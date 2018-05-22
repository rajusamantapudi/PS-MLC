import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from MF import ExplicitMF
np.random.seed(0)

def train_test_split(matrix):
    test = np.zeros(matrix.shape)
    train = matrix.copy()
    for user in range(matrix.shape[0]):
        test_guess = np.random.choice(matrix[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=True)
        train[user, test_guess] = 0.0
        test[user, test_guess] = matrix[user, test_guess]
        
    assert(np.all((train * test) == 0)) 
    return train, test



def get_best_model(train, test, model_type):
	iter_array = []
	regularizations = [0.01, 0.1, 1.0]
	latent_factors = []
	rate = 0.01
	if model_type == 'course_views':
		latent_factors = [20, 40, 80]
		iter_array = [1, 2, 5, 10, 25, 50, 100]
		rate = 0.001
	if model_type == 'tag_assessments':
		latent_factors = [5, 10, 20]
		iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
		rate = 0.01
	regularizations.sort()

	best_params = {}
	best_params['n_factors'] = latent_factors[0]
	best_params['reg'] = regularizations[0]
	best_params['n_iter'] = 0
	best_params['train_mse'] = np.inf
	best_params['test_mse'] = np.inf
	best_params['model'] = None

	for fact in latent_factors:
    		print 'Factors: {}'.format(fact)
    		for reg in regularizations:
        		print 'Regularization: {}'.format(reg)
        		MF_SGD = ExplicitMF(train, n_factors=fact, learning='sgd',\
                            user_fact_reg=reg, item_fact_reg=reg, \
                            user_bias_reg=reg, item_bias_reg=reg)
        		MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=rate)
        		min_idx = np.argmin(MF_SGD.test_mse)
        		if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
            			best_params['n_factors'] = fact
            			best_params['reg'] = reg
            			best_params['n_iter'] = iter_array[min_idx]
            			best_params['train_mse'] = MF_SGD.train_mse[min_idx]
            			best_params['test_mse'] = MF_SGD.test_mse[min_idx]
            			best_params['model'] = MF_SGD
            			print 'New optimal hyperparameters'
            			print pd.Series(best_params)

	return best_params['model']

def LV_course_views():
	names = ['user_handle', 'view_date', 'course_id', 'author_handle', 'level', 'view_time_seconds']
	df = pd.read_csv('../data/user_course_views.csv', names=names)
	# x = users
	# y = courses
	n_x = df.user_handle.unique().shape[0]
	n_y = df.course_id.unique().shape[0]
	x = df.user_handle.unique().tolist()
	y = df.course_id.unique().tolist()

	matrix = np.zeros((n_x, n_y))
	min_v = 10000
	max_v = -10000

	for row in df.itertuples():
    		if np.log(float(row[6])+1) > max_v:
        		max_v = np.log(float(row[6])+1)
    		if np.log(float(row[6])+1) < min_v:
        		min_v = np.log(float(row[6])+1)

	for row in df.itertuples():
    		matrix[x.index(row[1]), y.index(row[3])] = 4.0*(np.log(float(row[6])+1) - min_v)/(max_v-min_v) + 1

	train, test = train_test_split(matrix)
	best_model = get_best_model(train, test, 'course_views')
	user_vecs, user_bias = best_model.get_user_vecs()
	user_map_vecs = {}
	print_once=1
	for u in user_vecs:
		user_bias_list = [user_bias[u]]
		user_map_vecs[x[u]]=np.append(user_bias_list, user_vecs[u])
		if print_once==1:
			print user_bias_list
			print user_vecs[u]
			print user_map_vecs[x[u]]
			print_once+=1

	np.save("../models/uc.npy",user_map_vecs)
	return


def LV_tag_assessments():
	names = ['user_handle', 'assessment_tag', 'user_assessment_date', 'user_assessment_score']
	df = pd.read_csv('../data/user_assessment_scores.csv', names=names)
	# x = users
	# y = assessment_tags
	n_x = df.user_handle.unique().shape[0]
	n_y = df.assessment_tag.unique().shape[0]
	x = df.user_handle.unique().tolist()
	y = df.assessment_tag.unique().tolist()

	matrix = np.zeros((n_x, n_y))
	min_v = 10000
	max_v = -10000

	for row in df.itertuples():
    		if float(row[4]) > max_v:
        		max_v = float(row[4])
    		if float(row[4]) < min_v:
        		min_v = float(row[4])

	for row in df.itertuples():
    		matrix[x.index(row[1]), y.index(row[2])] = 4.0*(float(row[4]) - min_v)/(max_v-min_v) + 1

	train, test = train_test_split(matrix)
	best_model = get_best_model(train, test, 'tag_assessments')
	user_vecs, user_bias = best_model.get_user_vecs()
	user_map_vecs = {}
	print_once=1
	for u in user_vecs:
		user_bias_list = [user_bias[u]]
		user_map_vecs[x[u]]=np.append(user_bias_list, user_vecs[u])
		if print_once==1:
			print user_bias_list
			print user_vecs[u]
			print user_map_vecs[x[u]]
			print_once+=1

	np.save("../models/ua.npy",user_map_vecs)
	return


def calculate_similarity(data_items):
	data_sparse = sparse.csr_matrix(data_items)
	similarities = cosine_similarity(data_sparse.transpose())
	sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
	return sim

def get_user_vector(user, data, data_items, data_matrix):
	user_index = data[data.users == user].index.tolist()[0]
	known_user_likes = data_items.ix[user_index]
	known_user_likes = known_user_likes[known_user_likes >0].index.values
	user_rating_vector = data_items.ix[user_index]
	score = data_matrix.dot(user_rating_vector).div(data_matrix.sum(axis=1))
	return score

def dot(K, L):
	if len(K) != len(L):
		return 0
	return sum(i[0] * i[1] for i in zip(K, L))

def LV_tag_interests():
	names = ['user_handle', 'interest_tag', 'date_followed']
	df = pd.read_csv('../data/user_interests.csv', names=names)
	
	n_x = df.user_handle.unique().shape[0]
	n_y = df.interest_tag.unique().shape[0]
	x = df.user_handle.unique().tolist()
	y = df.interest_tag.unique().tolist()

	matrix = [ ([0] * n_y) for row in range(n_x) ]

	for index,row in df.iterrows():
		x_idx = x.index(row["user_handle"])
		y_idx = y.index(row["interest_tag"])
		matrix[x_idx][y_idx]=1

	header_str = ','.join(y)
	file_ui = open("../data/user-interests-new.csv", 'w')
	file_ui.write("users," + header_str +'\n')
	for i in range(len(x)):
		interest_str = ','.join(str(e) for e in matrix[i])        
		file_ui.write(str(x[i]) + "," + interest_str+'\n')
	file_ui.close()

	data = pd.read_csv('../data/user-interests-new.csv')
	data_items = data.drop('users', 1)
	users_list = data['users'].tolist()
	for i in range(len(users_list)):
		users_list[i] = int(users_list[i])

	magnitude = np.sqrt(np.square(data_items).sum(axis=1))
	data_items = data_items.divide(magnitude, axis='index')

	data_matrix = calculate_similarity(data_items)	

	user_score_vecs={}
	for u in users_list:
		user_score_vecs[u] = get_user_vector(u, data, data_items, data_matrix)

	np.save("../models/ui.npy",user_score_vecs)
        return


#LV_tag_assessments()
#LV_course_views()
LV_tag_interests()



