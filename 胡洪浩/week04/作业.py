# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


class TreeNode:
    def __init__(self, val, children=None):
        self.val = val
        self.children = children


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    """
    我们想顺序处理多个分支的任务，必须要有队列或者栈，python中采用列表模拟
    1. 构造多叉树
        采用两个队列并行添加节点的子节点，一个队列主要放切分后的句子作为下次的任务
        另一个队列主要放切分后的句子对应的节点，方便切分后添加该节点的分支孩子。
    2. 遍历多叉树
        三个容器：1. 节点栈，用于模仿递归的处理过程，遍历所有节点
                 2. 路径集合，用于存放到每个叶子节点的路径的步骤，采用栈的方式实现回溯效果
                 3. 目标集合，用于存放到达叶子节点的路径，设置叶子节点循环终止条件，判断完整路径
    """
    # TODO
    # 构造多叉树，,是一个寻找多叉树全部路径的问题
    if not sentence:
        return []
    root_node, tmp_que = TreeNode("", children=[]), [sentence]
    node_que = [root_node]
    while tmp_que:
        cur_node = node_que.pop(0)
        cur_words = tmp_que.pop(0)
        for i in range(len(cur_words)):
            cur_word = cur_words[:i + 1]
            if cur_word in Dict:
                new_node = TreeNode(cur_word, children=[])
                tmp_que.append(cur_words[i + 1:])
                cur_node.children.append(new_node)
                node_que.append(new_node)
        # print("tmp_children:", [i.val for i in cur_node.children if i.val] )
    # 遍历多叉树
    stack, path_st, result = [root_node], [root_node.val], []

    while stack:
        cur = stack.pop()
        path = path_st.pop()
        # 如果当前节点为叶子节点，添加路径到结果中
        if not cur.children:
            result.append(path)
        for i in cur.children:
            if i:
                stack.append(i)
                path_st.append(path + '/' + i.val)
    target = []
    for k in result:
        target.append(k.lstrip("/").split("/"))
    return target


# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]
print(all_cut(sentence, Dict))
