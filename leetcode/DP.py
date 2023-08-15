# -*- coding: utf-8 -*-
# @Time : 2023/8/14 18:40 
# @Author : gt1562
# @Email :tao.guan@aispeech.com


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        # 从上到下打印二叉树，即按层打印，又称为二叉树的广度优先搜索BFS
        # 通常借助队列的先入先出来实现
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        return res