#_*_ coding:utf-8 _*_

import numpy as np
from igraph import Graph as IGraph
import matplotlib.pyplot as plt


def loadData(filepath):
    try:
        with open(filepath,encoding='UTF-8') as filepath:
            lines = filepath.readlines()
            lineArray = np.array(lines)
    except IOError as e:
        print("您说读取的文件不存在")
    edges = []
    for i in range(len(lineArray)):
        wordArr = lineArray[i].split('|')
        for j in range(len(wordArr)):
            if wordArr[j].startswith('srcIp_dstIp_s'):
                word_split = wordArr[j].split(':')
                src_Dst = word_split[1].split(',')
                edges.append((src_Dst[0],src_Dst[1]))
    return edges

# 构造图
def bulidGraph(edges):
    return IGraph.TupleList(edges, directed=True, vertex_name_attr='name', edge_attrs=None, weights=False)

# 返回经过每个点的最短路径
def getAllPath(graph):
    names = graph.vs['name']
    for x in names:
        paths = graph.get_all_shortest_paths(x)
        for p in paths:
            print([names[m] for m in p])


if __name__ == '__main__':
    data = loadData(r"C:\Users\robin\Desktop\tda_20171217.dat")
    graph = bulidGraph(data)

    getAllPath(graph)
