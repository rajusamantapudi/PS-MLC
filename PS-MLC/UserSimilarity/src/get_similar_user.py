import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compress_vec():
	dc = np.load("../models/uc.npy")
	da = np.load("../models/ua.npy")
	di = np.load("../models/ui.npy")

	dc = dc.item()
	da = da.item()
	di = di.item()

	print len(dc)
	print len(da)
	print len(di)

	users_dc = dc.keys()
	users_da = da.keys()
	users_di = di.keys()

	users = users_dc + users_da + users_di
	users = list(set(users))
	
	s_curr = di[users[0]]
	for i in range(len(users)-1):
		s_next = di[users[i+1]]
		s_curr = pd.concat([s_curr, s_next], axis=1)
		print s_curr.shape
	print "-----------------"
	print s_curr.shape
	pca = PCA(n_components=50)
	pca.fit(s_curr)
	print pca.components_.shape
	print type(pca.components_)
	print pca.explained_variance_ratio_.cumsum()
	user_compress_vecs={}
	for i in range(len(users)):
		user_compress_vecs[users[i]]= pca.components_[:,i]
	np.save("../models/ui-c.npy",user_compress_vecs)
	
	return

def load_vecs():
	dc = np.load("../models/uc.npy")
	da = np.load("../models/ua.npy")
	#di = np.load("../models/ui.npy")
	di = np.load("../models/ui-c.npy")

	dc = dc.item()
	da = da.item()
	di = di.item()

	users_dc = dc.keys()
	users_da = da.keys()
	users_di = di.keys()

	users = users_dc + users_da + users_di
	users = set(users)
	#print dc[2266]
	#print "------"
	#print da[7860]
	#print "------"
	#print type(di[2266])
	#print di[2266][0]
	return users, dc, da, di

def getKey(item):
	return item[0]

def get_topn_similar_users(u, n=5):
	users, dc, da, di = load_vecs()
	if u not in users:
		print ("user handle not in the dataset")
		return
	sim_list = []
	sim=0
	sim_user = u
	for x in users:
		if x != u:
			sim_new = get_similarity(u, x, dc, da, di)
			#print sim_new
			#break
			sim_list.append([sim_new, x])
	sim_list = sorted(sim_list, key=getKey, reverse=True)
	print "-----------------------------"
	print "user \t | similarity score"
	print "-----------------------------"
	for sim_entry in sim_list[:n]:
		if sim_entry[0] > 0:
			print sim_entry[1], '\t |', sim_entry[0]
	return

def dot(K, L):
        if len(K) != len(L):
                return 0
        return sum(i[0] * i[1] for i in zip(K, L))

def get_similarity(u,x, dc, da, di):
	sim =0
	#print (x, u, "----")
	#print dc.keys()

	if x in dc and u in dc:
		sim=sim+dot(dc[x],dc[u])
		#print (sim,1)
	if x in da and u in da:
		sim=sim+dot(da[x],da[u])
		#print (sim,2)
	if x in di and u in di:
		sim=sim+dot(di[x],di[u])
		#print (sim,3)
	return sim

def main():
	u = raw_input('Enter User Handle to find similar users [current dataset user handle range is 1 to 10000]  : ')
	n = raw_input('How many similar users would you like to see [e.g. 5]  : ')
	try:
		u = int(u)
		n = int(n)
	except:
		print ("please provide correct user handle and number of similar users you would like to see")
		return
	if n <=0:
		print ("please provide a positive number for the number of similar users you would like to see")
		return
	get_topn_similar_users(u,n)
	return 

if __name__== "__main__":
  main()
  #compress_vec()
