import numpy as np 
#line = [1,2,3]
line = [2,3,1]
print(np.argmax(np.bincount(line)))

a = [[1,2,3,4],[4,5,6,7]]
b = [[1,3,2,4],[4,5,6,7]]
c = [[1,3,2,4],[4,5,6,7]]

# 获取两个列表不同元素的索引
'''
def compare(a,b):
	res = []
	if(len(a) == len(b)):
		for i in range(len(a)):
			if(a[i] == b[i]):
				pass
			else:
				res.append(i)
	return res

index_l = compare(a,b)
a_arr = np.array(a)
print(a_arr[index_l])

print(max(a_arr))
'''
a_arr = np.array(a)
b_arr = np.array(b)
c_arr = np.array(c)
print(a_arr+b_arr+c_arr)