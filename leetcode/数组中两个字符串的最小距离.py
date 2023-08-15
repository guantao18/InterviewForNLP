# -*- coding: utf-8 -*-
# @Time : 2023/8/1 18:39 
# @Author : gt1562
# @Email :tao.guan@aispeech.com
import math

strs=["1", "3", "3", "2", "3", "1"]
str1="1"
str2="2"
m,n = -1, -1
min_dis = 10000000
for i in range(len(strs)):
    if strs[i] == str1:
        m = i
    if strs[i] == str2:
        n = i
    if m != -1 and n != -1:
        min_dis = min(min_dis,abs(m-n))

print(min_dis)

