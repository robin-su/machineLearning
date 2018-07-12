# -*- coding:utf-8 -*-

import itertools

# list = list(itertools.permutations([0,1,2,3],2))
# list2 = []
#
# def test(list):
#     list2 = []
#     for i in range(len(list)):
#         if i >= len(list):
#             break
#         k,v = list[i]
#         if k == v:
#             list.remove(list[i])
#         elif (v,k) in list:
#             l = list.index((v,k))
#             list.remove(list[l])
#     return list
#
# # print(test(list))

print(list(itertools.combinations([0,1,2,3],2)))