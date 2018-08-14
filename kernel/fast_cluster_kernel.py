from sklearn.gaussian_process.kernels import Kernel as ParentKernel, StationaryKernelMixin, NormalizedKernelMixin
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.cluster import KMeans

# Random Forrest Kernel
class FastClusterKernel(StationaryKernelMixin, NormalizedKernelMixin, ParentKernel):


    def __init__(self):
        return
    # def add_Y(self, X, pred):
    # 	self.X = X
    # 	self.pred = pred
    # 	return

    # def generate_random(self, X):
    	
    # 	return depth

    def get_new_model(self, X, Y, corr_mat):
        num_cluster = np.random.randint(1, X.shape[0])+1
        kmeans = KMeans(n_clusters=num_cluster, init="random",
            max_iter=1).fit(X)
        x_pred = kmeans.predict(X)
        y_pred = kmeans.predict(Y)
        # print("num clusters ", num_cluster, X.shape, Y.shape, corr_mat.shape)
        for i in range(num_cluster):
            idx_ary_x = np.where(x_pred == i)[0]
            idx_ary_y = np.where(y_pred == i)[0]
            # print("X ", idx_ary_x)
            # print("Y ", idx_ary_y)
            for idx in idx_ary_y:
                # print("Idx ", idx, idx_ary_x)
                corr_mat[idx, idx_ary_x] += 1
    	return

    def run_model_n_times(self, X, Y):
    	N_x = len(X)
    	N_y = len(Y)

    	corr_mat = np.zeros((N_y, N_x))
    	M = 50#200
    	for i in range(M):
    		self.get_new_model(X, Y,  corr_mat)
    	corr_mat = corr_mat*1.0/M
    	# print("Correlation Mat ", corr_mat)
    	return corr_mat

    def __call__(self, X, Y=None):

        if Y is None:
     		corr_mat = self.run_model_n_times(X, X)
     	else:
     		corr_mat = self.run_model_n_times(Y, X) 
     	return corr_mat





