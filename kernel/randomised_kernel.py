from sklearn.gaussian_process.kernels import Kernel as ParentKernel, StationaryKernelMixin, NormalizedKernelMixin
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding

# Random Forrest Kernel
class RandForrestKernel(StationaryKernelMixin, NormalizedKernelMixin, ParentKernel):


    def __init__(self):
        return
    # def add_Y(self, X, pred):
    # 	self.X = X
    # 	self.pred = pred
    # 	return

    # def generate_random(self, X):
    	
    # 	return depth

    def get_new_model(self, X, Y, corr_mat):
    	max_depth = np.log2(X.shape[0])
    	# print("Max Depth ", max_depth)
    	depth = np.random.randint(max_depth)+1
    	# Create Model
    	hasher = RandomTreesEmbedding(n_estimators=1,
    		max_depth=depth)
    	hasher.fit(X)
    	x_transformed = hasher.transform(X)
    	x_trans_dense = x_transformed.todense()
    	y_transformed = hasher.transform(Y)
    	y_trans_dense = y_transformed.todense()
    	for i in range(x_trans_dense.shape[1]):
    		# print(x_trans_dense)
    		index_array_x = np.where(x_trans_dense[:,i] == 1.0)[0]

    		index_array_y = np.where(y_trans_dense[:,i] == 1.0)[0]
    		# print("Index array ", i, index_array)
    		for idx in index_array_y:
    			corr_mat[idx, index_array_x] += 1
    	# print(depth, corr_mat)
    	# print("Shape of the transformed ", X_transformed.shape, depth)
    	# print("hi ",np.where( != 0.0)[1])

    	# X_est = hasher.estimators_
    	# for estimator in X_est:
    	# 	pred = estimator.predict(X)
    	# 	print("Shape ", pred)

    	# reg.fit(self.X, self.pred)
    	
    	return

    def run_model_n_times(self, X, Y):
    	N_x = len(X)
    	N_y = len(Y)

    	corr_mat = np.zeros((N_y, N_x))
    	M = 200
    	for i in range(M):
    		self.get_new_model(X, Y,  corr_mat)
    	corr_mat = corr_mat*1.0/M
    	# print("Correlation Mat ", corr_mat.shape)
    	return corr_mat

    def __call__(self, X, Y=None):
    	
        # X = np.atleast_2d(X)
        # scale_factor = np.sum(X[0])
        # if Y is None:
        #     # print(scale_factor, X.shape)
        #     X = np.matrix(X)
        #     prod = X*X.T/scale_factor
        #     print("Only X ",X.shape, prod.shape)
        #     # return prod
        # else:
        #     # print(scale_factor, X.shape, Y.shape)
        #     X = np.matrix(X)
        #     Y = np.matrix(Y)
        #     prod = np.asarray(X*Y.T)/scale_factor
        #     # print("Both X, Y ", prod)
        #     print("Both X Y",X.shape, Y.shape, prod.shape)


        if Y is None:
     		corr_mat = self.run_model_n_times(X, X)
     	else:
     		corr_mat = self.run_model_n_times(Y, X) 
     	return corr_mat





