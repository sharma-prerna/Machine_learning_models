#machine learning GUI project
#Directory cd Documents\pdf\ML-prob-stat\Machine_learning_models
import streamlit as st
import k_means_cluster as KMC
import knn as KNN 
import matplotlib.pyplot as plt
import multi_linear_regression as Linear
import multi_logistic_regression as Logistic
import naive_bayes as NB
import deep_neural_network_model as DNN
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import load_dataset as LOAD
import seaborn as sns

def load_css(filename):
	with open(filename,"r") as file:
		st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)

def get_choice(choice):
	if choice=="Supervised Learning":
		ch = st.selectbox("Type of Supervised Learning",["Classification","Regression "])
		if ch=="Classification":
			algo = st.sidebar.selectbox("Type of Classification",["K Nearest Neighbors","Naive Bayes","Multilayer Perceptron","Logistic Regression"])
		else:
			algo = st.sidebar.selectbox("Type of Regression",["Linear Regression"])
		return algo

	else:
		ch = st.selectbox("Type of Clustering",["K Means Clustering"])
		return ch

def linear_regression_model(algo_type):
	options = ["Father Son Height Prediction","Graduate Admission Chance Prediction","Other"]
	ds = st.sidebar.selectbox("Choose Dataset (Choose Other to Upload External File)",options)
	split = st.sidebar.text_input("Enter test split percentage between 1 and 100 ( e.g. 30 or 40)")
	split_percent =None
	dataset = False
	status =None

	if split:
		try: 
			split_percent = int(split)
			if split_percent<1 or split_percent>100:
				st.stop()
		except:
			st.sidebar.error("SPLIT PERCENTAGE MUST BE AN INTEGER (BETWEEN 1 TO 100)")
			st.stop()

	if ds=="Father Son Height Prediction" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("father_son_height.csv",split_percent,"Regression")
	
	elif ds=="Graduate Admission Chance Prediction" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("Admission_predict.csv",split_percent,"Regression")
	
	elif ds=="Other" and split_percent:
		html = """<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
		<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
		<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li></ul>"""
		
		st.sidebar.markdown(html,unsafe_allow_html=True)
		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_external_csv_dataset(dataset,split_percent,"Regression")
	
	if status=="VALID DATASET" and dataset:
		
		st.sidebar.success(status)
		
		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test 

		st.sidebar.dataframe(train_data.head(10))
		st.sidebar.text(f"Training data size: {train_data.shape}")
		st.sidebar.text(f"Testing data size: {test_data.shape}")

		learning_rate = top.number_input("Learning Rate")
		num_iterations = top.slider("Iterations",1,5000)
		TB = top.button("Train")
		top.markdown("""<ul style="text-align:justify;"><li>Learning is decreasing as number of 
			iterations increases (for e.g. if steady learning rate = 0.1 & total iterations=100 
			then dynamic learning rate = 0.1/100</li> </ul>""",unsafe_allow_html=True)
		
		if TB:
			Y_train, Y_test = np.array(Y_train.copy()),np.array(Y_test.copy())
			Y_train = Y_train.reshape(1,train_data.shape[0])
			Y_test = Y_test.reshape(1,test_data.shape[0])

			X_train = Linear.normalize_data(X_train)
			X_test = Linear.normalize_data(X_test)

			w,b,costs = Linear.linear_regression(X_train.T,Y_train,learning_rate,num_iterations)
		
			Z_train = np.dot(w,X_train.T)+b
			MSE = np.sum((Z_train-Y_train)**2)/Y_train.shape[1]
			middle.text(f"Training Mean squared error is: {MSE}")

			Z_test = np.dot(w,X_test.T)+b
			MSE = np.sum((Z_test-Y_test)**2)/Y_test.shape[1]
			middle.text(f"Testing Mean squared error is: {MSE}")

			train_data["Prediction for "+ last]=np.round(np.squeeze(Z_train),2)
			test_data["Prediction for "+last]=np.round(np.squeeze(Z_test),2)

			middle.dataframe(train_data)
			plot(train_data,test_data,last,algo_type)
			plot_cost(costs)
			st.balloons()

	elif status is not None and dataset:
		st.sidebar.error(status)
		st.stop()


def k_nearest_neighbor_model(algo_type):
	options = ["Iris Flower Classification","Wine Quality Classification","Heart Disease Chances Prediction","Other"]
	ds = st.sidebar.selectbox("Choose Dataset (Choose Other to Upload External File)",options)
	split = st.sidebar.text_input("Enter test split percentage between 1 and 100 ( e.g. 30 or 40)")
	split_percent =None
	dataset=False
	status =None
	if split:
		try: 
			split_percent = int(split)
			if split_percent<1 or split_percent>100:
				st.stop()
		except:
			st.sidebar.error("SPLIT PERCENTAGE MUST BE AN INTEGER (BETWEEN 1 TO 100)")
			st.stop()

	if ds=="Iris Flower Classification" and split_percent:
		dataset =True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("Iris.csv",split_percent,"Classification")
	elif ds=="Wine Quality Classification" and split_percent:
		dataset =True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("wine.csv",split_percent,"Classification")
	elif ds=="Heart Disease Chances Prediction" and split_percent:
		dataset =True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("heart.csv",split_percent,"Classification")
	elif ds=="Other" and split_percent:
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
			<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
			<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_external_csv_dataset(dataset,split_percent,"Classification")
	
	if status=="VALID DATASET" and dataset:
		st.sidebar.success(status)
		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test

		st.sidebar.dataframe(train_data.head(10))
		st.sidebar.text(f"Training data size: {train_data.shape}")
		st.sidebar.text(f"Testing data size: {test_data.shape}")

		k = top.slider("K",1,20)
		TB = top.button("Train")
		
		if TB:

			#Z_test is the prediction
			Z_test = KNN.predict_knn(X_train,Y_train,X_test,k)
			accuracy = KNN.accuracy(Z_test,Y_test)
			middle.text("Testing Accuracy : {:.2%}".format(accuracy))
			#adding last column again
			test_data["Prediction for "+last]=np.squeeze(Z_test)
			
			middle.dataframe(test_data.head(10))
			plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
			plot(train_data,test_data,last,algo_type)
			st.balloons()
	
	elif status and dataset:
		st.sidebar.error(status)
		st.stop()

def naive_bayes_model(algo_type):
	ds = st.sidebar.selectbox("Choose Dataset (Choose Other to Upload External File)",["Iris Flower Classification","Wine Quality Classification","Heart Disease Chances Prediction","Other"])
	split = st.sidebar.text_input("Enter test split percentage between 1 and 100 ( e.g. 30 or 40)")
	split_percent =None
	dataset = False
	status =None
	if split:
		try: 
			split_percent = int(split)
			if split_percent<1 or split_percent>100:
				st.stop()
		except:
			st.sidebar.error("SPLIT PERCENTAGE MUST BE AN INTEGER (BETWEEN 1 TO 100)")
			st.stop()

	if ds=="Iris Flower Classification" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("Iris.csv",split_percent,"Classification")
	elif ds=="Wine Quality Classification" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("wine.csv",split_percent,"Classification")
	elif ds=="Heart Disease Chances Prediction" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("heart.csv",split_percent,"Classification")
	elif ds=="Other" and split_percent:
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
			<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
			<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_external_csv_dataset(dataset,split_percent,"Classification")
	
	if status=="VALID DATASET" and dataset:
		st.sidebar.success(status)
		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test 
		classes = list(train_data[last].unique())
		st.sidebar.dataframe(train_data.head(10))
		st.sidebar.text(f"Training data size: {train_data.shape}")
		st.sidebar.text(f"Testing data size: {test_data.shape}")

		TB = top.button("Train")
		if TB:

			marginal_prob,mean_std= NB.gaussian_naive_bayes_classiefier(train_data,classes,last)
			
			Z_train,accuracy = NB.predict(X_train,Y_train,mean_std,marginal_prob,classes)
			middle.text("Training Accuracy : {:.2%}".format(accuracy))

			Z_test,accuracy = NB.predict(X_test,Y_test,mean_std,marginal_prob,classes)
			middle.text("Testing Accuracy : {:.2%}".format(accuracy))

			test_data["Prediction for "+last]=np.squeeze(Z_test)
			train_data["Prediction for "+last]=np.squeeze(Z_train)

			middle.dataframe(train_data.head(10))
			plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
			st.balloons()
	
	elif status and dataset:
		st.sidebar.error(status)
		st.stop()



def logistic_regression_model(algo_type):
	top.info("Note: This Logistic Regression models performs only Binary classification")
	ds = st.sidebar.selectbox("Choose Dataset (Choose Other to Upload External File)",["Logic Gates","Wine Quality Classification","Heart Disease Chances Prediction","Other"])
	dataset = False
	status =None
	split_percent= None
	if ds=="Logic Gates":
		split=None
		split_percent= True
	else:
		split = st.sidebar.text_input("Enter test split percentage between 1 and 100 ( e.g. 30 or 40)")
	
	if split:
		try: 
			split_percent = int(split)
			if split_percent<1 or split_percent>100:
				st.stop()
		except:
			st.sidebar.error("SPLIT PERCENTAGE MUST BE AN INTEGER (BETWEEN 1 TO 100)")
			st.stop()

	if ds=="Logic Gates" and split_percent:
		dataset = True
		gate = top.selectbox("Choose Gate",["AND","OR","NAND","NOR"])
		n = top.selectbox("Number of Inputs",range(2,7))
		status,X_train,Y_train,X_test,Y_test,last = LOAD.create_boolean_dataset(n,gate)
	
	elif ds=="Wine Quality Classification" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("wine.csv",split_percent,"Classification")
	
	elif ds=="Heart Disease Chances Prediction" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("heart.csv",split_percent,"Classification")
	
	elif ds=="Other" and split_percent:
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
			<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
			<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_external_csv_dataset(dataset,split_percent,"Classification")
	
	if dataset and status=="VALID DATASET":

		st.sidebar.success(status)
		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
	
		classes = list(test_data[last].unique())
		st.sidebar.text(f"Classes are: {classes}")
		st.sidebar.dataframe(test_data.head(10))
		if len(classes)>2:
			top.error("More than two classes found in dataset. Can't classify")
			st.stop()

		st.sidebar.text(f"Training data size: {train_data.shape}")
		st.sidebar.text(f"Testing data size: {test_data.shape}")

		learning_rate = top.number_input("Learning Rate")
		num_iterations = top.slider("Iterations",1,5000)
		TB = top.button("Train")
		top.markdown("""<ul style="text-align:justify;"><li>Learning is decreasing as number of iterations increases (for e.g. if steady learning rate = 0.1
				; total iterantions=100 then dynamic learning rate = 0.1/100</li> </ul>""",unsafe_allow_html=True)
		if TB and len(classes)<=2:
			
			train_data[last], test_data[last] = pd.Categorical(train_data[last]), pd.Categorical(test_data[last])
			cat_map = dict(enumerate(test_data[last].cat.categories))
			train_data[last], test_data[last] = train_data[last].cat.codes, test_data[last].cat.codes
			
			middle.text(cat_map)

			Y_train, Y_test = np.array(train_data[last].copy()), np.array(test_data[last].copy())
			Y_train,Y_test = Y_train.reshape(1,train_data.shape[0]), Y_test.reshape(1,test_data.shape[0])

			#if ouptut vector contains string
			X_train = Logistic.normalize_data(X_train)
			X_test = Logistic.normalize_data(X_test)
		
			w,b,costs = Logistic.logistic_regression(X_train.T,Y_train,learning_rate,num_iterations)

			Z_train,accuracy,A_train = Logistic.predict(X_train.T,Y_train,w,b)
			middle.text("Training Accuracy : {:.2%}".format(accuracy))

			Z_test,accuracy,A_test = Logistic.predict(X_test.T,Y_test,w,b)
			middle.text("Testing Accuracy : {:.2%}".format(accuracy))

			train_data["Prediction for "+last] = np.squeeze(Z_train)
			test_data["Prediction for "+last] = np.squeeze(Z_test)
			
			middle.dataframe(test_data)
			plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
			plot_decision_boundary(test_data,w,b,last)
			plot_cost(costs)
			st.balloons()
	elif status and dataset:
		st.sidebar.error(status)
		st.stop()


def multilayer_perceptron_model(algo_type):
	top.markdown("""<p style ="font-style:italic;">Multilayer perceptron is a feedforward neural network that consists three types of layers
		given below:</p> <ul><li>Input layer</li><li>Hidden layer</li><li>Output layer</li></ul><p style ="font-style:italic;">Input layer size (number of perceptrons in input layer)
		must be equal to the number of features in dataset. Output layer size must be equal to the number of distinct classes in
		the output vector.<br>Note: Hidden layer might have more than one layer in itself</p>""",unsafe_allow_html=True)
	top.info("Note: This Multilayer perceptron models performs only Binary classification")

	ds = st.sidebar.selectbox("Type of Dataset",["XOR Gate","Can vs Non cat image classification","Wine Quality Classification","Heart Disease Chances Prediction","Other"])
	dataset=False
	status =None
	if ds=="XOR Gate" or ds=="Can vs Non cat image classification":
		split=None
		split_percent=True
	else:
		split = st.sidebar.text_input("Enter test split percentage between 1 and 100 ( e.g. 30 or 40)")
		split_percent =None
	
	if split:
		try: 
			split_percent = int(split)
			if split_percent<1 or split_percent>100:
				st.stop()
		except:
			st.sidebar.error("SPLIT PERCENTAGE MUST BE AN INTEGER (BETWEEN 1 TO 100)")
			st.stop()
	
	if ds=="Can vs Non cat image classification" and split_percent:
		dataset=True
		status,X_train,Y_train,X_test,Y_test= LOAD.load_cat_noncat_dataset()
		last = "category"
		#flatten the data i.e change 12 ,12 , 3 array to  (1,12*12*3)
		m=X_train.shape[0]
		n=X_test.shape[0]
		X_train=X_train.reshape(m,-1)
		X_test=X_test.reshape(n,-1)
		Y_train=Y_train.reshape(1,m)
		Y_test=Y_test.reshape(1,n)
		classes=["Cat","NonCat"]
		#normalizing the array
		X_train=X_train/255
		X_test=X_test/255

		layers=None

	elif ds=="XOR Gate" and split_percent:
		dataset=True
		n = top.slider("No. of Inputs",2,7)
		status,X_train,Y_train,X_test,Y_test,last = LOAD.create_boolean_dataset(n,"XOR")

	elif ds=="Wine Quality Classification" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("wine.csv",split_percent,"Classification")
	
	elif ds=="Heart Disease Chances Prediction" and split_percent:
		dataset = True
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_internal_csv_dataset("heart.csv",split_percent,"Classification")
	
	elif ds=="Other" and split_percent:
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
		<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
		<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,X_test,Y_test,last = LOAD.load_external_csv_dataset(dataset,split_percent,"Classification")
	
	if status=="VALID DATASET" and dataset:
		st.sidebar.success(status)
		train_data, test_data = X_train.copy(), X_test.copy()

		if ds!="Can vs Non cat image classification":
			train_data[last], test_data[last] = Y_train, Y_test
			classes = list(test_data[last].unique())
			st.sidebar.text(f"Classes are: {classes}")
			st.sidebar.dataframe(train_data.head(10))
			if len(classes)>2:
				top.error("More than two classes found in dataset. Can't classify")
				st.stop()

		st.sidebar.text(f"Training data size: {train_data.shape}")
		st.sidebar.text(f"Testing data size: {test_data.shape}")

		nl = top.text_input("Total number of layers in hidden layer")
		no_l = None
		try:
			no_l= int(nl)
			if no_l<0:
				top.error("Number of layers can not be less than 0")
				st.stop()
		except:
			top.error("Number of layers must be positive integer")
			st.stop()
		
		if no_l is not None:
			layers = [0]*(no_l+2)
			layers[0] = X_train.shape[1]
			layers[-1] = 1

			top.text(f"Input layer size : {layers[0]}")
			
			for i in range(1,no_l+1):
				layers[i]=top.text_input(f"Hidden layer{i} size : ")
				try:
					layers[i]= int(layers[i])
					if layers[i]<0:
						top.error("Size of layer can not be less than 0")
						break
				except:
					top.error("Size of layers must be positive integer")
					break

			
			top.text(f"Output layer size: {layers[-1]}")
			input_activation = top.selectbox("Activation function for first n-1 layers",["relu","sigmoid","tanh"])
			output_activation = top.selectbox("Activation function for the ouput layer",["sigmoid","tanh"])
			learning_rate = top.number_input("Learning Rate")
			num_iterations = top.slider("Iterations",1,10001)
			
			top.markdown("""<ul style="text-align:justify;"><li>Default activation functions: Relu and Sigmoid respectively (recommended)</li>
				<li>Learning rate is decreasing as number of iterations increases. (for e.g. if steady learning rate = 0.1
				; total iterantions=100 then dynamic learning rate = 0.1/100</li> </ul>""",unsafe_allow_html=True)
			TB = top.button("Train")
			
			if TB:

				if ds!="Can vs Non cat image classification":
					train_data[last], test_data[last] = pd.Categorical(train_data[last]), pd.Categorical(test_data[last])
					cat_map = dict(enumerate(test_data[last].cat.categories))
					train_data[last], test_data[last] = train_data[last].cat.codes, test_data[last].cat.codes
					
					middle.text(cat_map)

					Y_train, Y_test = np.array(train_data[last].copy()), np.array(test_data[last].copy())
					Y_train,Y_test = Y_train.reshape(1,train_data.shape[0]), Y_test.reshape(1,test_data.shape[0])

				X_train,X_test=DNN.normalize_data(X_train,X_test)

				parameters,costs= DNN.deep_neural_network(layers,learning_rate,num_iterations,X_train.T,Y_train,input_activation,output_activation)

				Z_train,accuracy,A_train = DNN.predict(X_train.T,Y_train,parameters,input_activation,output_activation)
				middle.text("Training Accuracy : {:.2%}".format(accuracy))

				Z_test,accuracy,A_test = DNN.predict(X_test.T,Y_test,parameters,input_activation,output_activation)
				middle.text("Testing Accuracy : {:.2%}".format(accuracy))
				
				if ds!="Can vs Non cat image classification":
					train_data["Prediction for "+last] = np.squeeze(Z_train)
					test_data["Prediction for "+last] = np.squeeze(Z_test)
					
					middle.dataframe(test_data)
					plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
				
				else:
					actual = list(np.squeeze(Y_test))
					predicted = list(np.squeeze(Z_test))
					for i in range(len(actual)):
						actual[i]= "NonCat" if actual[i]==0 else "Cat"
					for i in range(len(predicted)):
						predicted[i]= "NonCat" if predicted[i]==0 else "Cat"

					fig, ax = plt.subplots(1,3)
					for i in range(3):
						img = test_data[i]
						ax[i].imshow(img.reshape(64,64,3))

					middle.text(f"Actual Category : {actual[:3]}")
					middle.text(f"Predicted Category: {predicted[:3]}")
					middle.pyplot(fig)
					data = {'Actual':actual,'Prediction':predicted}
					df = pd.DataFrame(data)
					plot_confusion_matrix(df['Actual'],df['Prediction'])
				
				plot_cost(costs)
				st.balloons()

	elif status and dataset:
		st.sidebar.error(status)
		st.stop()

def k_means_clustering_model(algo_type):
	ds = st.sidebar.selectbox("Type of Dataset",["Iris Flower Dataset","Wine Quality Dataset","Other"])
	dataset = False
	status =None
	
	if ds=="Iris Flower Dataset":
		dataset = True
		status,X_train,Y_train,_,_,last = LOAD.load_internal_csv_dataset("Iris.csv",0,"Clustering")
	
	elif ds=="Wine Quality Dataset":
		dataset = True
		status,X_train,Y_train,_,_,last = LOAD.load_internal_csv_dataset("wine.csv",0,"Clustering")
	
	elif ds=="Other":
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
			<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
			<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,_,_,last = LOAD.load_external_csv_dataset(dataset,0,"Clustering")
	
	if dataset and status=="VALID DATASET":
		st.sidebar.success(status)
		label ="Seeds"
		X_train[label] = list(map(str,range(1,X_train.shape[0]+1)))
		
		st.sidebar.dataframe(X_train)
		st.sidebar.text(f"Training data size: {X_train.shape}")

		k = top.slider(f"Value of K ",1,100)
		
		cluster_names = top.text_input("Enter valid {} cluster seed names:".format(k))
		cluster_names= cluster_names.strip().split()
		TB = top.button("Train")
		
		if TB:
			clustered_data = KMC.k_means_clustering(X_train,cluster_names,label)
			middle.dataframe(clustered_data)
			clusters = clustered_data["cluster"].unique()
			middle.write(f"{k} distinct clusters are : {clusters}")
			plot(clustered_data,None,"cluster",algo_type)
			st.balloons()

def plot(train_data,test_data,last,type):
	
	if type=="Linear Regression":
		fig,ax = plt.subplots(figsize=(7,4))
		cols = list(train_data.columns)
		x1 = np.array(train_data[cols[0]])
		y1 = np.array(train_data[cols[-2]])
		x2 = np.array(test_data[cols[0]])
		y2 = np.array(test_data[cols[-2]])

		#computing regression line
		z = np.squeeze(test_data[cols[-1]])
		x_min, z_min = x2.min(),z.min()
		x_max, z_max = x2.max(),z.max()
		
		x_line = np.linspace(x_min,x_max,num=len(np.squeeze(x2)))
		z_line = np.linspace(start=z_min,stop=z_max,num=len(np.squeeze(z)))
		
		ax.scatter(x1,y1,color='blue',label='Training data points',edgecolor='white')
		ax.scatter(x2,y2,color='green',label='Testing data points',edgecolor='white')

		ax.plot(x_line,z_line,color='g',label='Regression Line')
		ax.set_title('Linear Regression')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[-2])
		ax.legend(loc='upper right')
		ax.set_facecolor("aliceblue")
		bottom.pyplot(fig)

	elif type=="K Nearest Neighbors":
		cols = list(train_data.columns)
		fig,ax = plt.subplots(figsize=(7,4))
		sns.scatterplot(x=cols[0],y=cols[1],data=train_data,hue=last)
		sns.scatterplot(x=cols[0],y=cols[1],s=150,data=test_data,color='black',marker='*',legend=False,label='Test Instances')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[1])
		ax.set_facecolor("aliceblue")
		ax.legend(loc='upper right')
		bottom.pyplot(fig)

	elif type=="K Means Clustering":
		cols = list(train_data.columns)
		fig,ax = plt.subplots()
		sns.scatterplot(x=cols[0],y=cols[1],hue=last,data=train_data)	
		ax.set_title('K Means Clustering')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[1])
		ax.set_facecolor("lavender")
		ax.legend()
		bottom.pyplot(fig)


def plot_confusion_matrix(actual,predicted):
	classes = list(actual.unique())
	confusion_matrix = pd.crosstab(actual,predicted,rownames=['Actual'],colnames=['Predicted'],margins=True)
	#st.dataframe(confusion_matrix)
	fig, ax = plt.subplots(figsize=(7,4))
	sns.heatmap(confusion_matrix,annot=True)
	confusion.pyplot(fig)

	if len(classes)<=2:
		try:
			TP = confusion_matrix[classes[0]][classes[0]]
		except:
			TP = 0

		try:
			TN = confusion_matrix[classes[1]][classes[1]]
		except:
			TN = 0

		try:
			FP = confusion_matrix[classes[1]][classes[0]]
		except:
			FP = 0

		try:
			FN = confusion_matrix[classes[0]][classes[1]]
		except:
			FN = 0

		recall = TP/(TP+FN) if TP or FN else np.nan
		sensitivity = TP/(TP+FN) if TP or FN else np.nan
		specificity = TN/(FP+TN) if TN or FP else np.nan
		precision = TP/(TP+FP) if TP or FP else np.nan
		Fscore = 2/((1/recall) + (1/precision)) if recall and precision else np.nan

		confusion.text(f"TN: {TN} FP:{FP} TP:{TP} FN:{FN}")
		data = {'Measure':["Sensitivity","Specificity","Precision","Recall","F1 Score"],'Value':[sensitivity,specificity,precision,recall,Fscore]}
		df = pd.DataFrame(data)
		confusion.table(df.style.set_properties(**{'text-align':'justify'}))


def plot_cost(costs):
	fig, ax = plt.subplots(figsize=(7,4))
	ax.plot(costs)
	ax.set_xlabel("Iterations")
	ax.set_ylabel("Cost")
	ax.set_facecolor("aliceblue")
	ax.grid()
	ax.set_title("Cost Reduction Graph")
	cost_graph.pyplot(fig)

def plot_decision_boundary(data,W,b,last):
	cols = list(data.columns)
	#normlize the data
	data[cols[0]], data[cols[1]] = Logistic.normalize_data(data[cols[0]]), Logistic.normalize_data(data[cols[1]])
	X, Y = data[cols[0]], data[cols[1]]
	x1 = np.arange(X.min()-0.1, X.max()+0.1, 0.1)
	x2 = np.arange(Y.min()-0.1, Y.max()+0.1, 0.1)
	
	w1, w2 = W[:,0],W[:,1]
	x, y = np.meshgrid(x1,x2)
	f = lambda x, y: Logistic.sigmoid(x*w1+ y*w2 + b)
	z= f(x,y)

	fig , ax = plt.subplots(figsize=(7,4))
	sns.scatterplot(x=cols[0],y=cols[1],data=data,s=70,hue=last,palette='dark')
	ax.contourf(x,y,z,alpha=0.6,levels=0,cmap='inferno')
	ax.set_xlabel(cols[0])
	ax.set_ylabel(cols[1])
	ax.legend()
	ax.set_title("Decision Boundary")
	decision.pyplot(fig)


if __name__=="__main__":
	#load the css file
	load_css("gui.css")
	st.markdown("""<h1 style="color:#152238;text-align:center;font-weight:bold;font-style:italic">Machine Learning Models</h1>""",unsafe_allow_html=True)
	#st.balloons()

	#creating three different partitions of main division
	top = st.beta_expander("Parameters")
	middle = st.beta_expander("Output")
	
	#sidebar options
	with st.sidebar:

		st.markdown("""<h3 style="color:#152238;text-align:justify;font-style:italic;"><span style="font-size:30px;">
			Machino!</span> is a web application. One can execute and analyze any machine learning algorithm with the help of it.
		  	For more information read its documentation</h3>""",unsafe_allow_html=True)
		
		choice = st.selectbox("Type of learning",["Supervised Learning","Unsupervised Learning"])
		algo = get_choice(choice)
		
	#get the algorithm type

	if algo=="Linear Regression":
		bottom = st.beta_expander("Graph Plot")
		cost_graph = st.beta_expander("Cost Reduction Plot")
		linear_regression_model(algo)

	elif algo=="K Nearest Neighbors":
		bottom = st.beta_expander("Graph Plot")
		confusion = st.beta_expander("Confusion Matrix")
		k_nearest_neighbor_model(algo)
		
	elif algo=="Naive Bayes":
		confusion = st.beta_expander("Confusion Matrix")
		naive_bayes_model(algo)

	elif algo=="Logistic Regression":
		confusion = st.beta_expander("Confusion Matrix")
		decision = st.beta_expander("Decision Boundary Plot")
		cost_graph = st.beta_expander("Cost Reduction Plot")
		logistic_regression_model(algo)
				
	elif algo=="Multilayer Perceptron":
		confusion = st.beta_expander("Confusion Matrix")
		cost_graph = st.beta_expander("Cost Reduction Plot")
		multilayer_perceptron_model(algo)
			
	else:
		bottom = st.beta_expander("Graph Plot")
		k_means_clustering_model(algo)
		