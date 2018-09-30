import pandas as pd 
import numpy as np 

svm_df = pd.read_csv('../result/result-tfidf-feature-prob.csv')
lg_df = pd.read_csv('../result/result-tfidf-feature-prob.csv')
nb_df = pd.read_csv('../result/result-tfidf-feature-prob.csv')

def series2arr(series):
	res = []
	for row in series:
		res.append(np.array(eval(row)))
	return np.array(res)


# Series
svm_prob_arr = series2arr(svm_df['prob'])
lg_prob_arr = series2arr(lg_df['prob'])
nb_prob_arr = series2arr(nb_df['prob'])



final_prob_arr = svm_prob_arr+lg_prob_arr+nb_prob_arr



result = pd.DataFrame(columns=['prob'])

result['prob'] = final_prob_arr.tolist()


print(result[:1])


prob_l = [e[0][1] for e in a]
print(len)