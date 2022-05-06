# from collections import Counter


# def sol2(str):
#     c = Counter(str)
#     res = ''
#     while True:
#         if c['W'] > 0:
#             res += 'W'
#             c['W'] -= 1
#         if c['D'] > 0:
#             res += 'D'
#             c['D'] -= 1
#         if c['L'] > 0:
#             res += 'L'
#             c['L'] -= 1
#         if len(res) == len(str):
#             break
#     print(res)


# def sol3(matrix):
#     m = len(matrix)
#     n = len(matrix[0])
#     border = []
#     for i in range(m):
#         for j in range(n):
#             if(matrix[i][j] == 1):
#                 border.append((i, j))
#     res = 0
#     p = [0, 2]
#     deltas = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
#     for b in border:
#         x, y = b[0], b[1]
#         cur_len = 1
#         for delta in deltas:
#             while 0 <= x < m and 0 <= y < n:
#                 x += delta[0]
#                 y += delta[1]
#                 if 0 <= x < m and 0 <= y < n and matrix[x][y] == p[cur_len % 2]:
#                     cur_len += 1
#                 else:
#                     break

#             if x == 0 or y == 0 or x == m-1 or y == n-1:
#                 res = max(res, cur_len)
#             x, y = b[0], b[1]
#             cur_len = 1
#     print(res)


# def sol4(a, b, queries):

#     d = Counter(a)
#     l = []
#     for q in queries:
#         if q[0] == 0:
#             d[a[q[1]]] -= 1
#             a[q[1]] = q[2]
#             if q[2] not in d:
#                 d.setdefault(q[2], 1)
#             else:
#                 d[q[2]] += 1
#         elif q[0] == 1:
#             res = 0
#             for num_b in b:
#                 res += d[q[1] - num_b]
#             l.append(res)
#     print(l)


# # a = [2, 3]
# # b = [1, 2, 2]
# # q = [[1, 4], [0, 0, 3], [1, 5]]
# # sol4(a, b, q)
# # matrix = [
# #     [2, 1, 2, 2, 0],
# #     [0, 2, 0, 2, 2],
# #     [0, 0, 0, 0, 0],
# #     [0, 0, 2, 2, 2],
# #     [2, 1, 0, 2, 1],
# #     [2, 2, 0, 0, 2]
# # ]

# # m2 = [[0, 0, 1, 2],
# #       [0, 2, 2, 2],
# #       [2, 1, 0, 1]]

# # m3 = [[1, 1, 1, 1, 1, 0],
# #       [1, 1, 1, 1, 2, 1],
# #       [1, 1, 1, 1, 1, 1],
# #       [1, 1, 2, 1, 1, 1],
# #       [1, 0, 1, 1, 1, 1],
# #       [2, 1, 1, 1, 1, 1]]

# # m4 = [
# #     [1, 1, 1],
# #     [1, 1, 1],
# #     [1, 1, 1]
# # ]
# # sol3(matrix)
# # sol3(m2)
# # sol3(m3)
# # sol3(m4)


# def codesignal2(words, variableName):
#     def camel_case_split(string):
#         bldrs = [[string[0].upper()]]
#         for c in string[1:]:
#             if bldrs[-1][-1].islower() and c.isupper():
#                 bldrs.append([c])
#             else:
#                 bldrs[-1].append(c)
#         return [''.join(bldr) for bldr in bldrs]
#     d = set(words)
#     split_words = camel_case_split(variableName)
#     print(split_words)
#     for w in split_words:
#         w = w.lower()
#         print(w)
#         if w not in d:
#             return False
#     return True
# words = ['is','valid', 'right']
# variableName = 'IsValid'
# # print(codesignal2(words, variableName))


# def s4(s):
#     c = Counter(s)
#     candidate = set()
#     for k,v in c.items():
#         if v == 1:
#             candidate.add(k)
#     if not candidate:
#         return '_'
#     for ch in s:
#         if ch in candidate:
#             return ch
#     return '_'

# print(s4("abacabad"))
# print(s4("abacabaabacaba"))
import models
import torch
import numpy as np
from PIL import Image


# def test_nyu_depth_dataset():
#     path_to_train = 'dataset/nyu_depth_v2/train/1385_depth.png'
#     im = Image.open(path_to_train)
#     i = np.asarray(im)
#     # depth_arr = (i / 1000.0)
#     print(i.shape)
#     im.show()
# test_nyu_depth_dataset()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    multi_inputdevice = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = models.UnetAdaptiveBinsPointcloud.build(
        n_bins=128).to(multi_inputdevice)

    pointcloud = torch.rand(2, 3, 512 ).type(torch.float64).to(multi_inputdevice)
    x = torch.rand(2, 3, 416, 544).to(multi_inputdevice)

    bin_edges, pred = model(x, pointcloud)
    
    print(bin_edges.shape, pred.shape)
