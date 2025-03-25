#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project ：KnowledgeGrapheBuild 
@File    ：neo4j_practice.py
@IDE     ：PyCharm 
@Author  ：Guyuelegend
@Date    ：2025/3/24 11:19 
"""
from py2neo import Graph, Subgraph
from py2neo import Node, Relationship, Path

"""
有关于知识图谱中构建图数据库的练习，py2neo库与neo4j图数据库的使用
"""

# 首先我们需要连接数据库 创建图对象
graphe_url = "http://localhost:7474/"
# graph = Graph(graphe_url, username="neo4j", password="123456")  # 旧版本
# graph = Graph("bolt:" + graphe_url.strip("http:"), auth=("neo4j", "123456"))
graph = Graph(graphe_url, auth=("neo4j", "123456"))
print(graph)

# 想重新建立空白图对象，就先清空已有的所有节点
graph.delete_all()

# 避免中途出现错误，可以创建一个工作流，在此工作流上所有任务全部可以执行才能更新整个图数据库
graph = graph.begin(readonly=False)  # 参数看你希不希望只读只查看不修改

# Node定义节点以及操作=>创建一个实体=>对应的数据形式  label,key=value....
new_node1 = Node("人物", name="张无忌", identity="男主角")
new_node2 = Node("人物", name="张翠山", identity="起因角色")
new_node3 = Node("人物", name="殷素素", identity="起因角色")
new_node4 = Node("人物", name="赵敏", identity="女主角")
new_node5 = Node("人物", name="张三丰", identity="战力天花板")
new_node6 = Node("人物", name="小昭", identity="波斯圣女")
new_node7 = Node("人物", name="谢逊", identity="金毛狮王")
graph.create(new_node1)
graph.create(new_node2)
graph.create(new_node3)
graph.create(new_node4)
graph.create(new_node5)
graph.create(new_node6)
graph.create(new_node7)
print(new_node1)

# Relationship建立实体之间的关系=>数据形式  node1 relation node2
n1_to_n2 = Relationship(new_node1, "父亲", new_node2, ship="血缘")
n2_to_n5 = Relationship(new_node2, "师傅", new_node5)
n1_to_n3 = Relationship(new_node1, "母亲", new_node3)
n1_to_n4 = Relationship(new_node1, "爱人", new_node4)
n1_to_n7 = Relationship(new_node1, "义父", new_node7)
n6_to_n1 = Relationship(new_node6, "喜欢", new_node1)
n2_to_n3 = Relationship(new_node2, "妻子", new_node3)
n3_to_n2 = Relationship(new_node3, "夫君", new_node2)
graph.create(n1_to_n2)
graph.create(n2_to_n5)
graph.create(n1_to_n3)
graph.create(n1_to_n4)
graph.create(n1_to_n7)
graph.create(n6_to_n1)
graph.create(n2_to_n3)
graph.create(n3_to_n2)

# 也可以按照路径的方式一次性创建一条关系路径,形如path（节点，-关系->,节点，<-关系-,节点）默认是正向关系，也可自定义关系
node1, node2, node3 = Node(name="老大"), Node(name="老二"), Node(name="老三")
path_1 = Path(node1, "弟弟", node2, "妹妹", node3)
graph.create(path_1)

graph.commit()
