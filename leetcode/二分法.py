# -*- coding: utf-8 -*-
# @Time : 2023/8/10 22:34 
# @Author : gt1562
# @Email :tao.guan@aispeech.com

"""
二分法解题思路：
    左闭右开
    左闭右闭

"""
from typing import List
class Solution:

    def mySqrt(self,x):
        """
        1. 计算x的平方根，不可用内置函数pow()或者x**0.5
        偏移量的精度越高越趋近真实值,偏移量不能取整，如果取整的话会陷入死循环.
        :param x:
        :return:
        """
        l = 0
        r = x
        ans = 0.
        while l<=r:
            mid = (l + r) // 2
            if mid * mid <= x:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans

        #2.更快的解法：牛顿迭代法

    def searchRange(self,nums: List[int],target: int):
        """
        2.非递减数组，找target的起始位置，没有返回-1
        :param nums:
        :param target:
        :return:
        """
        #左闭右开：循环不变量：更新的l和r也需要在集合内
        def lower_bound(nums,target):
            #找左边界原理：遇到目标值通过减少右界继续往左压缩
            l = 0
            r = len(nums)
            while l < r:
                mid = l + (r-l) // 2  #避免溢出int的最大范围
                if nums[mid] >= target:
                    r = mid
                else:
                    l = mid + 1
            return l

        def upper_bound(nums,target):
            #找右边界原理：遇到目标值通过增加左界继续往右压缩
            l = 0
            r = len(nums)
            while l < r:
                mid = l + (r-l) // 2  #避免溢出int的最大范围
                if nums[mid] > target:
                    r = mid
                else:
                    l = mid + 1
            return l

        l = lower_bound(nums,target)
        r = upper_bound(nums,target)-1
        # 处理不在数组的情况，不在l会增长到长度
        if l<=r and r<=len(nums) and nums[l]==target and nums[r]== target:
            return [l,r]
        return [-1,-1]

if __name__ == "__main__":
    solution = Solution()
    result = solution.searchRange([5,6,7,7,8,9],7)
    print(result)
