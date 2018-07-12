
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community

'''
    加载数据的图
'''
def loadData(filepath):
    try:
        with open(filepath,encoding='UTF-8') as filepath:
            lines = filepath.readlines()
            print(len(lines))
            lineArray = np.array(lines)
    except IOError as e:
        print("您说读取的文件不存在")
    edges = []
    nodes = []
    for i in range(len(lineArray)):
        wordArr = lineArray[i].split('|')
        for j in range(len(wordArr)):
            if wordArr[j].startswith('srcIp_dstIp_s'):
                word_split = wordArr[j].split(':')
                src_Dst = word_split[1].split(',')
                edges.append((src_Dst[0],src_Dst[1]))
                nodes.append(src_Dst[0])
                nodes.append(src_Dst[1])
    return edges,nodes

'''
    构造图
    nodes:类型list了,表示节点list
    edges:类型list(tuple),表示边
'''
def builGraph(nodes,edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

'''
    绘制图
'''
def drawGraph(graph):
    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    nx.draw(graph, alpha=0.8, with_labels=True, node_size=200)
    # nx.draw(graph, alpha=0.8, node_color='r', node_size=250)
    plt.show()

'''
    计算pageRange值,使用pageRange算法，可以知道该ip被其他ip关连的程度，
    能从一定意义上说明某个IP，或者服务器存在较大隐患
'''
def calculatePagerank(graph,topk=10):
   pr = nx.pagerank(graph,alpha=0.85)
   pgvs = []
   for node,pageRankValue in pr.items():
       pgvs.append({"name": node, "pg":pageRankValue})
   topk_ = sorted(pgvs, key=lambda k: k['pg'], reverse=True)[:topk]
   return topk_

'''
    社区发现算法:可以定位出，在哪些节点社区，黑客运动的紧密度
'''
def communityDesc(graph):
    comp = community.girvan_newman(graph)
    t = tuple(sorted(c) for c in next(comp))
    for i in range(len(t)):
        print(t[i],end='\n')

# def communityByKlique(graph,k=3):
#     pos = nx.spring_layout(graph)
#     plt.clf()
#     klist = list(community.k_clique_communities(graph, k))
#     nx.draw(graph, pos=pos, with_labels=False)
#     nx.draw(graph, pos=pos, nodelist=klist[0], node_color='b')
#     nx.draw(graph, pos=pos, nodelist=klist[1], node_color='y')

'''
    计算网络中的节点的介数中心性，并进行排序输出
    可以计算出黑客攻击的关键节点
'''
def topNBetweeness(graph,topk=5):
    score = nx.betweenness_centrality(graph)
    score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    for tm in score[:topk]:
        print(tm,end='\n')


'''
    获取所有节点的最短路径
'''
def fetchAllShortestPath(graph):
    shortPathList = nx.all_pairs_shortest_path(graph)
    for path in shortPathList:
        print(path)
    return shortPathList

'''
    获取源IP的最短路径(会列出所有的路径)
'''
def attackRoute(graph,sourceIp):
    dict = nx.single_source_shortest_path(graph, sourceIp)
    listAttack = []
    routeTupleList = []
    for (d, x) in dict.items():
        if len(x) > 1:
            listAttack.extend(x)
            routeTupleList.append(tuple(x))
    listAttack = list(set(listAttack))
    routeTupleList = list(set(routeTupleList))
    return builGraph(listAttack,routeTupleList)


if __name__ == '__main__':
    edges,nodes = loadData(r"C:\Users\robin\Desktop\tda_20171217.dat")
    graph = builGraph(nodes,edges)

    drawGraph(graph)
    # #获取最短路径
    # fetchAllShortestPath(graph)
    # # pagerank = calculatePagerank(graph)
    # # for v in pagerank:
    # #     print(v,end='\n')
    # # 攻击路行图
    # route = attackRoute(graph, '13.114.72.101')
    # drawGraph(route)

    # communityDesc(graph)
    # topNBetweeness(graph)
