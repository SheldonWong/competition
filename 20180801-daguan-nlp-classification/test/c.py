class_weight = {14: 6740,
                  3: 8313,
                  12: 5326,
                  13: 7907,
                  1: 5375,
                  10: 4963,
                  19: 5524,
                  18: 7066,
                  7: 3038,
                  9: 7675,
                  4: 3824,
                  17: 3094,
                  2: 2901,
                  8: 6972,
                  6: 6888,
                  11: 3571,
                  15: 7511,
                  5: 2369,
                  16: 3220}
k = sum(list(class_weight.values()))

d = {}
for item in class_weight.items():
   print(item)
   d[item[0]-1] = item[1] / k
print(d)