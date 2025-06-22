"""
基本操作list, set, dict, tuple, str, int, float

list

| 操作               | 内容      | 計算量        |
| ---------------- | ------- | ---------- |
| `a.append(x)`    | 末尾に追加   | O(1)（平均）   |
| `a.pop()`        | 末尾から削除  | O(1)       |
| `a.insert(i, x)` | i番目に挿入  | O(N)       |
| `a.pop(i)`       | i番目を削除  | O(N)       |
| `a.remove(x)`    | 値xを削除   | O(N)       |
| `a.sort()`       | 昇順に並び替え | O(N log N) |
| `a.reverse()`    | リストの反転  | O(N)       |
| `x in a`         | 存在確認    | O(N)       |

set

| 操作             | 内容               | 計算量      |
| -------------- | ---------------- | -------- |
| `s.add(x)`     | 要素を追加            | O(1)（平均） |
| `s.discard(x)` | 要素を削除（なければ何もしない） | O(1)（平均） |
| `s.remove(x)`  | 要素を削除（なければエラー）   | O(1)（平均） |
| `x in s`       | 存在確認             | O(1)（平均） |
| `len(s)`       | 要素数取得            | O(1)     |

dict

| 操作               | 内容              | 計算量      |
| ---------------- | --------------- | -------- |
| `d[key] = value` | 追加・更新           | O(1)（平均） |
| `d.get(key, 0)`  | 存在しない場合のデフォルト取得 | O(1)     |
| `del d[key]`     | キーを削除           | O(1)     |
| `key in d`       | キー存在確認          | O(1)     |
| `len(d)`         | キーの数            | O(1)     |

tuple

| 操作              | 内容       | 計算量  |
| --------------- | -------- | ---- |
| `t = (1, 2, 3)` | 宣言       | O(1) |
| `t[i]`          | i番目にアクセス | O(1) |
| `(x, y) = t`    | アンパック    | O(1) |

str

| 操作                | 内容       | 計算量  |
| ----------------- | -------- | ---- |
| `s[i]`            | i文字目アクセス | O(1) |
| `s[::-1]`         | 反転       | O(N) |
| `s.split()`       | 空白で分割    | O(N) |
| `s.replace(a, b)` | 文字列の置換   | O(N) |
| `"a" in s`        | 部分一致     | O(N) |

int

| 操作               | 内容         | 計算量         |
| ---------------- | ---------- | ----------- |
| `str(x)`         | 文字列化       | O(D)（D: 桁数） |
| `int(s)`         | 数値化        | O(D)        |
| `bin(x)`         | 2進数文字列化    | O(log x)    |
| `divmod(x, y)`   | 商と余り同時取得   | O(1)        |
| `abs(x)`         | 絶対値        | O(1)        |
| `pow(x, y, mod)` | べき乗（mod付き） | O(log y)    |

float

| 操作              | 内容         | 計算量  |
| --------------- | ---------- | ---- |
| `round(f, k)`   | 小数第k位で四捨五入 | O(1) |
| `int(f)`        | 小数切り捨て     | O(1) |
| `math.floor(f)` | 小さい整数へ     | O(1) |
| `math.ceil(f)`  | 大きい整数へ     | O(1) |

変換操作

| 目的        | 方法              | 計算量  |
| --------- | --------------- | ---- |
| 数 → 文字列   | `str(x)`        | O(D) |
| 文字列 → 数   | `int(s)`        | O(D) |
| リスト → 集合  | `set(lst)`      | O(N) |
| 集合 → リスト  | `list(set)`     | O(N) |
| 文字列 → リスト | `list("abc")`   | O(N) |
| リスト → 文字列 | `"".join(list)` | O(N) |


"""
from collections import defaultdict,deque,Counter
"""
--------------------------------------

# defaultdict
# キーが存在しないときに自動で初期値を作る辞書
dd = defaultdict(int)  # int() → 0が初期値
dd['apple'] += 1      # 存在しなくてもエラーにならず0+1になる
print(dd['apple'])    # 1

# deque
# 両端キュー（高速な両端挿入・削除が可能）
dq = deque([1, 2, 3])
dq.append(4)         # 右端に追加
dq.appendleft(0)     # 左端に追加
print(dq)            # deque([0, 1, 2, 3, 4])
dq.pop()             # 右端削除
dq.popleft()         # 左端削除

# Counter
# 要素の出現回数を数える辞書
a = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
counter = Counter(a)
print(counter['apple'])  # 3
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

--------------------------------------
"""
from decimal import Decimal, ROUND_HALF_UP
"""
print(bin(255))                 # 10進数 -> 2進数
print(hex(255))                 # 10進数 -> 16進数

print(int('0b11111111', 2))     # 2進数 -> 10進数
print(int('0xff', 16))          # 16進数 -> 10進数
"""
from functools import lru_cache
"""
#再帰関数でメモ化するために使うかも(memo={}とかでいいけど、オーバーロードするときに便利かも)
@lru_cache(maxsize=None)
def fibo(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)

print(fibo(35))
"""
from heapq import heapify, heappop, heappush
#dijkstra法で必要。
from bisect import bisect, bisect_left, bisect_right
"""
--------------------------------------

# bisect（挿入位置を探す関数）の使い方

a = [1, 2, 4, 4, 5]

# bisect_left：値xを挿入する際、左側の挿入位置（同じ値があれば一番左）
pos_left = bisect_left(a, 4)  # 2

# bisect_right（またはbisect）：値xを挿入する際、右側の挿入位置（同じ値があれば一番右）
pos_right = bisect_right(a, 4)  # 4

print(pos_left, pos_right)

# 使いどころ：ソート済み配列に値を効率的に挿入したり、範囲内の個数を数えたりできる

# 例：配列内の4の個数を数える
count_4 = bisect_right(a, 4) - bisect_left(a, 4)  # 2

print(count_4)

--------------------------------------
"""
from typing import Generic, Iterable, Iterator, TypeVar
from itertools import accumulate, permutations, combinations, product,groupby
"""
--------------------------------------
b = list(accumulate(a))

a = [1, 1, 2, 3, 3, 3, 1, 2, 2]

--------------------------------------
counter = Counter(a)

for key, value in counter.items():
    print(key, value)
--------------------------------------
a = product([0,1],repeat=3)

(0, 0, 0)
(0, 0, 1)
(0, 1, 0)
(0, 1, 1)
(1, 0, 0)
(1, 0, 1)
(1, 1, 0)
(1, 1, 1)


colors = ['red', 'blue']
sizes = ['S', 'M', 'L']

for c, s in product(colors, sizes):
    print(c, s)

# 出力例
# red S
# red M
# red L
# blue S
# blue M
# blue L

"""
import string
"""
print(string.ascii_lowercase)
print(string.ascii_uppercase)
print(string.ascii_letters)
print(string.digits)
print(string.hexdigits)

abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
0123456789abcdefABCDEF
"""
import math
"""
丸め
| 関数              | 内容          | 計算量  | 備考               |
| --------------- | ----------- | ---- | ---------------- |
| `math.floor(x)` | ⌊x⌋ 下方向に丸める | O(1) | `int(x)` との違いに注意 |
| `math.ceil(x)`  | ⌈x⌉ 上方向に丸める | O(1) | 切り上げ             |
| `math.trunc(x)` | 0方向へ丸める     | O(1) | 小数点以下を切り捨て       |
| `round(x)`      | 四捨五入（偶数丸め）  | O(1) | Pythonの組み込みと同様   |
指数対数pow関数
| 関数               | 内容          | 計算量      | 備考              |
| ---------------- | ----------- | -------- | --------------- |
| `math.sqrt(x)`   | √x（平方根）     | O(1)     | `x ** 0.5` より正確 |
| `math.pow(x, y)` | x^y（浮動小数）   | O(1)     | `**` より遅く精度高    |
| `pow(x, y, m)`   | x^y mod m   | O(log y) | 組み込み関数（整数）      |
| `math.log(x)`    | ln(x)（自然対数） | O(1)     | x > 0           |
| `math.log10(x)`  | log₁₀(x)    | O(1)     | 10進対数           |
| `math.log2(x)`   | log₂(x)     | O(1)     | x のビット長に関係      |
整数処理
| 関数               | 内容      | 計算量             | 備考                           |
| ---------------- | ------- | --------------- | ---------------------------- |
| `math.gcd(x, y)` | 最大公約数   | O(log min(x,y)) | 3項以上は `functools.reduce` と併用 |
| `math.lcm(x, y)` | 最小公倍数   | O(log min(x,y)) | Python 3.9～                  |
| `math.isqrt(x)`  | √xの整数部分 | O(log x)        | 小数を返さない                      |


"""
import copy
#b = a[:]より、 b = copy.deepcopy(a)のほうが利便性が高いが、B = [a[:] for a in A]の方がましなので使わない
import sys
#sys.setrecursionlimit(10**9)で用いる。再帰関数の呼び出し上限んを増やす。
sys.set_int_max_str_digits(0)
#多倍長整数で殴るための
from pprint import pprint
import re

"""
UnionFind
基本操作はこんなもん
uf = UnionFind(5)        # 0〜4の5要素
uf.union(0, 1)           # 0と1を結合
uf.union(3, 4)           # 3と4を結合
print(uf.same(0, 1))     # True
print(uf.same(1, 2))     # False
print(uf.size(0))        # 2
print(uf.group_count())  # 3（[0,1], [2], [3,4]）

応用編
uf.find(x)	要素 x の属する集合の根（代表）を返す	uf.find(3)
uf.union(x, y)	x, y の属する集合を統合する	uf.union(1, 4)
uf.same(x, y)	x と y が同じ集合に属するか	uf.same(2, 3)
uf.size(x)	x の属する集合のサイズ	uf.size(1)
uf.roots()	すべての集合の代表（根）をリストで返す	uf.roots()
uf.group_count()	現在の集合の個数	uf.group_count()
uf.members(x)	x の属する集合に含まれる全要素	uf.members(0)
uf.all_group_members()	すべての集合の構成要素を {根: [要素リスト]} の形で返す	uf.all_group_members()

"""
class UnionFind():
    #「uf = UnionFind(頂点の数)」で初期化
    
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n
 
    def find(self, x): #uf.find(x)
        #要素xが属するグループの根を返す
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
 
    def union(self, x, y): #uf.union(x, y)
        #要素xが属するグループと要素yが属するグループを併合
        x = self.find(x)
        y = self.find(y)
 
        if x == y:
            return
 
        if self.parents[x] > self.parents[y]:
            x, y = y, x
 
        self.parents[x] += self.parents[y]
        self.parents[y] = x
 
    def size(self, x): #uf.size(x)
        #要素xが属するグループのサイズ(要素数)を返す
        return -self.parents[self.find(x)]
 
    def same(self, x, y): #uf.same(x,y)
        #要素x,yが同じグループに属するかどうかを返す
        return self.find(x) == self.find(y)
 
    def members(self, x): #uf.members(x)
        #要素xが属するグループに属する要素をリストで返す
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]
 
    def roots(self): #uf.roots()
        #根となっている要素すべてをリストで返す
        return [i for i, x in enumerate(self.parents) if x < 0]
 
    def group_count(self): #uf.group_count()
        #グループの数を返す
        return len(self.roots())
 
    def all_group_members(self): #uf.all_group_members()
        #{ルート要素 : [そのグループに含まれる要素のリスト], ...}のdefaultdictを返す
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members
 
    def __str__(self):
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())


"""
bt = BinaryTrie()

bt.insert(5)     # 集合に 5 を追加
bt.insert(7)     # 集合に 7 を追加
bt.insert(5)     # 5 をもう一度追加（multiset）
bt.insert(9)     # 集合に 9 を追加

print(bt.count(5))      # → 2
print(bt.size())        # → 4（全体の要素数）

print(bt.kth_elm(1))    # → 5（1番目）
print(bt.kth_elm(2))    # → 5（2番目）
print(bt.kth_elm(3))    # → 7（3番目）
print(bt.kth_elm(4))    # → 9（4番目）

print(bt.lower_bound(6))  # → 3（6以上で最小は 7、3番目）

bt.erase(5)         # 5を1つ削除（残り1個）
print(bt.count(5))  # → 1

"""
class BinaryTrie:
    def __init__(self, max_query=2*10**5, bitlen=30):
        n = max_query * bitlen
        self.nodes = [-1] * (2 * n)
        self.cnt = [0] * n
        self.id = 0
        self.bitlen = bitlen

    def size(self):
        return self.cnt[0]

    def count(self,x): #xの個数
        pt = 0
        for i in range(self.bitlen-1,-1,-1):
            y = x>>i&1
            if self.nodes[2*pt+y] == -1:
                return 0
            pt = self.nodes[2*pt+y]
        return self.cnt[pt]

    def insert(self,x): #xの挿入
        pt = 0
        for i in range(self.bitlen-1,-1,-1):
            y = x>>i&1
            if self.nodes[2*pt+y] == -1:
                self.id += 1
                self.nodes[2*pt+y] = self.id
            self.cnt[pt] += 1
            pt = self.nodes[2*pt+y]
        self.cnt[pt] += 1


    def erase(self,x): #xの削除、xが存在しないときは何もしない
        if self.count(x) == 0:
            return
        pt = 0
        for i in range(self.bitlen-1,-1,-1):
            y = x>>i&1
            self.cnt[pt] -= 1
            pt = self.nodes[2*pt+y]
        self.cnt[pt] -= 1

    
    def kth_elm(self,x): #昇順x番目の値(1-indexed)
        assert 1 <= x <= self.size()
        pt, ans = 0, 0
        for i in range(self.bitlen-1,-1,-1):
            ans <<= 1
            if self.nodes[2*pt] != -1 and self.cnt[self.nodes[2*pt]] > 0:
                if self.cnt[self.nodes[2*pt]] >= x:
                    pt = self.nodes[2*pt]
                else:
                    x -= self.cnt[self.nodes[2*pt]]
                    pt = self.nodes[2*pt+1]
                    ans += 1
            else:
                pt = self.nodes[2*pt+1]
                ans += 1
        return ans

    def lower_bound(self,x): #x以上の最小要素が昇順何番目か(1-indexed)、x以上の要素がない時はsize+1を返す
        pt, ans = 0, 1
        for i in range(self.bitlen-1,-1,-1):
            if pt == -1: break
            if x>>i&1 and self.nodes[2*pt] != -1:
                ans += self.cnt[self.nodes[2*pt]]
            pt = self.nodes[2*pt+(x>>i&1)]
        return ans

T = TypeVar('T')

class SortedMultiset(Generic[T]):
    BUCKET_RATIO = 16
    SPLIT_RATIO = 24
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedMultiset from iterable. / O(N) if sorted / O(N log N)"
        a = list(a)
        n = self.size = len(a)
        if any(a[i] > a[i + 1] for i in range(n - 1)):
            a.sort()
        num_bucket = int(math.ceil(math.sqrt(n / self.BUCKET_RATIO)))
        self.a = [a[n * i // num_bucket : n * (i + 1) // num_bucket] for i in range(num_bucket)]

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __eq__(self, other) -> bool:
        return list(self) == list(other)
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedMultiset" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _position(self, x: T) -> tuple[list[T], int, int]:
        "return the bucket, index of the bucket and position in which x should be. self must not be empty."
        for i, a in enumerate(self.a):
            if x <= a[-1]: break
        return (a, i, bisect_left(a, x))

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a, _, i = self._position(x)
        return i != len(a) and a[i] == x

    def count(self, x: T) -> int:
        "Count the number of x."
        return self.index_right(x) - self.index(x)

    def add(self, x: T) -> None:
        "Add an element. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a, b, i = self._position(x)
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.SPLIT_RATIO:
            mid = len(a) >> 1
            self.a[b:b+1] = [a[:mid], a[mid:]]
    
    def _pop(self, a: list[T], b: int, i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a: del self.a[b]
        return ans

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a, b, i = self._position(x)
        if i == len(a) or a[i] != x: return False
        self._pop(a, b, i)
        return True

    def lt(self, x: T) -> T | None:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> T | None:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> T | None:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> T | None:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, i: int) -> T:
        "Return the i-th element."
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0: return a[i]
        else:
            for a in self.a:
                if i < len(a): return a[i]
                i -= len(a)
        raise IndexError
    
    def pop(self, i: int = -1) -> T:
        "Pop and return the i-th element."
        if i < 0:
            for b, a in enumerate(reversed(self.a)):
                i += len(a)
                if i >= 0: return self._pop(a, ~b, i)
        else:
            for b, a in enumerate(self.a):
                if i < len(a): return self._pop(a, b, i)
                i -= len(a)
        raise IndexError

    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans

class SortedSet(Generic[T]):
    BUCKET_RATIO = 16
    SPLIT_RATIO = 24
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        a = list(a)
        n = len(a)
        if any(a[i] > a[i + 1] for i in range(n - 1)):
            a.sort()
        if any(a[i] >= a[i + 1] for i in range(n - 1)):
            a, b = [], a
            for x in b:
                if not a or a[-1] != x:
                    a.append(x)
        n = self.size = len(a)
        num_bucket = int(math.ceil(math.sqrt(n / self.BUCKET_RATIO)))
        self.a = [a[n * i // num_bucket : n * (i + 1) // num_bucket] for i in range(num_bucket)]

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __eq__(self, other) -> bool:
        return list(self) == list(other)
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedSet" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _position(self, x: T) -> tuple[list[T], int, int]:
        "return the bucket, index of the bucket and position in which x should be. self must not be empty."
        for i, a in enumerate(self.a):
            if x <= a[-1]: break
        return (a, i, bisect_left(a, x))

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a, _, i = self._position(x)
        return i != len(a) and a[i] == x

    def add(self, x: T) -> bool:
        "Add an element and return True if added. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return True
        a, b, i = self._position(x)
        if i != len(a) and a[i] == x: return False
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.SPLIT_RATIO:
            mid = len(a) >> 1
            self.a[b:b+1] = [a[:mid], a[mid:]]
        return True
    
    def _pop(self, a: list[T], b: int, i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a: del self.a[b]
        return ans

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a, b, i = self._position(x)
        if i == len(a) or a[i] != x: return False
        self._pop(a, b, i)
        return True
    
    def lt(self, x: T) -> T | None:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> T | None:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> T | None:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> T | None:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, i: int) -> T:
        "Return the i-th element."
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0: return a[i]
        else:
            for a in self.a:
                if i < len(a): return a[i]
                i -= len(a)
        raise IndexError
    
    def pop(self, i: int = -1) -> T:
        "Pop and return the i-th element."
        if i < 0:
            for b, a in enumerate(reversed(self.a)):
                i += len(a)
                if i >= 0: return self._pop(a, ~b, i)
        else:
            for b, a in enumerate(self.a):
                if i < len(a): return self._pop(a, b, i)
                i -= len(a)
        raise IndexError
    
    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans

#n進数に直す。base_to(5,2)->[1,0,1]
def base_to(num, base):
    if num == 0: return ['0']
    res_list = []
    while num:
        res_list.append(str(num % base))
        num //= base
    return res_list[::-1]
#10進数に直す。base_to(1001,2)-> 8
def base_from(s, base):
    # sが文字列ならそのまま int() に渡す
    if isinstance(s, str):
        return int(s, base)
    # sがリストなら桁ごとに計算する
    elif isinstance(s, (list, tuple)):
        res = 0
        for d in s:
            res = res * base + int(d)
        return res
    else:
        raise TypeError("base_from の引数は str または list/tuple でなければなりません")


def clamp(num,smallest,largest): #numがsmallest以下ならsmallestに、largest以上ならlargestに調整
    return max(smallest,min(num,largest))

def count_digit(num): #整数numの桁数
    return math.floor(math.log10(num)) + 1 if num > 0 else 1

def Manhattan_distance(ax,ay,bx,by): #座標a(ax,ay)とb(bx,by)の2点間距離
    return math.sqrt(Decimal((ax-bx)**2 + (ay-by)**2))

def divisor(x): #整数xの約数をすべて入れたリスト
    divisors = []
    sqrt_x = int(x ** 0.5)
    for i in range(1, sqrt_x + 1):
        if x % i == 0:
            divisors.append(i)
            if i != x // i:
                divisors.append(x // i)
    return divisors

def check_in_grid(height,width,i,j): #(i,j)が height x widthのグリッドの中の点か確認
    return ((0 <= i < height) and (0 <= j < width))

def check_intersection(a,b,c,d, flg_edge=False):#数直線上の線分abと線分cdの共通部分があるかどうかチェック(flg_edgeがTrueなら端点のみの共有を含む)
    if flg_edge:
        return (max(a,c) <= min(b,d))
    else:
        return (max(a,c) < min(b,d))

def is_over_180degree(ax,ay,bx,by): #ベクトルa(ax,ay)とベクトルb(bx,by)の角度(aから反時計回りに)が180°より大きければ1、180°ちょうどなら2
    if ax*by - bx*ay < 0:
        return 1
    elif ax*by - bx*ay == 0:
        return 2
    return 0

def is_prime(i): #iが素数かの判定
    if i <= 1:
        return False
    for j in range(2, int(i**0.5) + 1):
        if i % j == 0:
            return False
    return True
"""
A = [3, 1, 5, 2, 6, 4, 7]
この中で、以下のような「増加する部分列」が取れる

[3, 5, 6, 7]

[1, 2, 4, 7] ← これが返る

[1, 2, 6, 7]

[1, 5, 6, 7]
などなど。
"""
def longest_increasing_subsequence(A, INF=10**9): #配列Aの最長増加部分列LISのリスト、計算量O(N*logN)
    dp = [INF for _ in A]
    b = [-1 for _ in A]
    for i in range(len(A)):
        idx = bisect_left(dp, A[i])
        dp[idx] = A[i]
        b[i] = idx + 1
    l = bisect_left(dp, INF)
    seq = [0 for i in range(l)]
    for i in range(len(A)-1, -1, -1):
        if b[i] == l:
            l -= 1
            seq[l] = A[i]
    return seq
"""
print(my_round(127,  0))   # → 130（1の位で四捨五入）
print(my_round(127,  2))   # → 100（100の位で四捨五入）
print(my_round(3.146, -2)) # → 3.15（小数第2位）
print(my_round(3.144, -2)) # → 3.14
"""
def my_round(num, d):#偶数丸めではない四捨五入、dは四捨五入の桁数(ex:0は1の位、2は100の位、-2は0.01の位)
    if d <= 0:
        return Decimal(str(num)).quantize(Decimal(str(10**d)), rounding=ROUND_HALF_UP)
    else:
        p = Decimal(str(num)).quantize(Decimal("1E" + str(d)), rounding=ROUND_HALF_UP)
        return p.quantize(Decimal(1))

def list_turn_right(l): #2次元配列lを時計回りに90度回転
    return list(zip(*l[::-1]))
def list_turn_left(l):  # 反時計回りに90度回転
    return list(zip(*l))[::-1]

def pascal_triangle(n): #n段のパスカルの三角形(list, 計算量O(n**2))
    res = []
    for i in range(1,n+1):
        if i == 1:
            tmp = [1]
        elif i == 2:
            tmp = [1,1]
        else:
            tmp = []
            for j in range(i):
                if j == 0 or j == i-1:
                    tmp.append(1)
                else:
                    tmp.append(res[i-2][j-1] + res[i-2][j])
        res.append(tmp)
    return res

def pow_x(x, n): #xの0乗～n乗までのリスト
    List_pow = [1]
    for _ in range(n):
        List_pow.append(x * List_pow[-1])
    return List_pow

def prime_factorize(num): #numを素因数分解したリスト
    factors = []
    while num % 2 == 0:
        factors.append(2)
        num //= 2
    f = 3
    while f * f <= num:
        while num % f == 0:
            factors.append(f)
            num //= f
        f += 2
    if num > 1:
        factors.append(num)
    
    return factors
"""
連長圧縮
s = "aaabbaaaccc"
rle = run_length_encoding(s)
print(rle)  # [['a', 3], ['b', 2], ['a', 3], ['c', 3]]
"""
def run_length_encoding(str_a: str): #連長圧縮、「ある文字がいくつ連続しているか」を順番に集めたリスト
    res = [[key,len(list(group))] for key,group in groupby(str_a)]
    return res
#素数のリスト
def Sieve_of_Eratosthenes(num): #num以下の数へのエラトステネスの篩(sortedlist)、計算量O(n*loglogn)
    res = [True] * (num + 1)
    res[0] = res[1] = False
    for i in range(2, int(num**0.5) + 1):
        if res[i]:
            for j in range(i*i, num + 1, i):
                res[j] = False
    return [i for i in range(num + 1) if res[i]]

def triangle_area(ax,ay,bx,by,cx,cy):#a(ax,ay),b(bx,by),c(x,cy)の3点からなる三角形の面積
    return abs((bx-ax)*(cy-ay) - (cx-ax)*(by-ay)) / 2
"""
セグメントツリー
加算だったら
st = SegTree(
    op = lambda a, b: a + b,       # 加算
    e = lambda: 0,                 # 加算の単位元（0）
    n = N,                         # 要素数
    v = A                          # 初期配列（任意）
)
XORだったら
st = SegTree(
    op = lambda a, b: a ^ b,       # XOR（ビットごとの排他的論理和）
    e = lambda: 0,                 # XORの単位元（0）
    n = N,
    v = A
)

"""
class SegTree:
    def __init__(self, op, e, n, v=None):
        self._n = n
        self._op = op
        self._e = e
        self._log = (n - 1).bit_length()
        self._size = 1 << self._log
        self._d = [self._e()] * (self._size << 1)
        if v is not None:
            for i in range(self._n):
                self._d[self._size + i] = v[i]
            for i in range(self._size - 1, 0, -1):
                self._d[i] = self._op(self._d[i << 1], self._d[i << 1 | 1])
    
    def set(self, p, x):
        p += self._size
        self._d[p] = x
        while p:
            self._d[p >> 1] = self._op(self._d[p], self._d[p ^ 1])
            p >>= 1
    
    def get(self, p):
        return self._d[p + self._size]

    def prod(self, l, r):
        sml, smr = self._e(), self._e()
        l += self._size
        r += self._size
        while l < r:
            if l & 1:
                sml = self._op(sml, self._d[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self._op(self._d[r], smr)
            l >>= 1
            r >>= 1
        return self._op(sml, smr)
    
    def all_prod(self):
        return self._d[1]
def input(): return sys.stdin.readline().rstrip('\n')
if 1: #入力系
    #def i(): return input()
    #---1つの文字列の受け取り
    def i(): return int(input())
    #---1つの整数の受け取り
    def mi(n = 0): return map(lambda x: int(x)+n, input().split())
    #---スペースで区切られた複数の整数をそれぞれ+nして受け取り
    def li(n = 0): return list(map(lambda x: int(x)+n, input().split()))
    #---スペースで区切られた複数の整数をそれぞれ+nしてリストで受け取り
"""
ダイクストラ法
# ABC などの最短距離だけ問う問題に最適
dist = dijkstra(edges, n)
print(dist[goal])

# ルートを復元しなきゃいけない場合
path, dist = dijkstra(edges, n, Goal)
print("→".join(map(str, path)))

"""
def dijkstra(edges, num_node):
    """ 経路の表現
            [終点, 辺の値]
            A, B, C, D, ... → 0, 1, 2, ...とする """
    node = [float('inf')] * num_node    #スタート地点以外の値は∞で初期化
    node[0] = 0     #スタートは0で初期化

    node_name = []
    heappush(node_name, [0, 0])

    while len(node_name) > 0:
        #ヒープから取り出し
        _, min_point = heappop(node_name)
        
        #経路の要素を各変数に格納することで，視覚的に見やすくする
        for factor in edges[min_point]:
            goal = factor[0]   #終点
            cost  = factor[1]   #コスト

            #更新条件
            if node[min_point] + cost < node[goal]:
                node[goal] = node[min_point] + cost     #更新
                #ヒープに登録
                heappush(node_name, [node[min_point] + cost, goal])

    return node
def dijkstra(edges, num_node, Goal):
    """ 経路の表現
            [終点, 辺の値]
            A, B, C, D, ... → 0, 1, 2, ...とする """
    node = [float('inf')] * num_node    #スタート地点以外の値は∞で初期化
    node[0] = 0     #スタートは0で初期化

    node_name = []
    heappush(node_name, [0, [0]])

    while len(node_name) > 0:
        #ヒープから取り出し
        _, min_point = heappop(node_name)
        last = min_point[-1]
        if last == Goal:
            return min_point, node  #道順とコストを出力させている
        
        #経路の要素を各変数に格納することで，視覚的に見やすくする
        for factor in edges[last]:
            goal = factor[0]   #終点
            cost  = factor[1]   #コスト

            #更新条件
            if node[last] + cost < node[goal]:
                node[goal] = node[last] + cost     #更新
                #ヒープに登録
                heappush(node_name, [node[last] + cost, min_point + [goal]])

    return []


def Yes(ok): return print("Yes" if ok else "No")
#---変数"ok"がTrueなら"Yes"、Falseなら"No"を出力

#再帰関数の回数上限突破
sys.setrecursionlimit(10**7)

#import string
ALPHABET = list(string.ascii_uppercase) #大文字アルファベットのリスト(["A", "B", "C", ....])
alphabet = list(string.ascii_lowercase) #小文字アルファベットのリスト(["a", "b", "c", ....])
Numbers = list(string.digits) #1桁の数字のリスト(["0","1","2", ....])(各要素はstr)

#座標の移動　[0:4]で上右下左の4方向、[0:8]で斜めを加えた時計回り8方向
dir_x = [0,1,0,-1,1,1,-1,-1]
dir_y = [1,0,-1,0,1,-1,-1,1]
dx_dy = [[0,1],[1,0],[0,-1],[-1,0]]
INF = float('inf')
MOD1 = 998244353
MOD2 = 10**9+7

#latestupdate 20240704
#-----------------------------------------
#-----------------------------------------

