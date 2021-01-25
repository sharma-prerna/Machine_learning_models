#LOAD DATASETS OF DIFFERENT TYPES

import numpy as np
import h5py
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_integer_dtype
#from pandas.api.types import is_bool_dtype
from pandas.api.types import is_float_dtype


def create_boolean_dataset(n,choice):
	data_range = np.array(range(0,2**n),dtype=int)
	train_input = np.empty((np.power(2,n),n),dtype=int)

	for num in data_range:
		train_input[num]= binary_conversion(num,n)


	if choice=="OR":
		train_output = np.ones(2**n,dtype=int)
		train_output[0]= 0

	elif choice=="NAND":
		train_output = np.ones(2**n,dtype=int)
		train_output[-1] = 0

	elif choice=="NOR":
		train_output = np.zeros(2**n,dtype=int)
		train_output[0]= 1

	elif choice=="AND":
		train_output = np.zeros(2**n,dtype=int)
		train_output[-1] = 1

	elif choice=="XOR":
		train_output = np.ones(2**n,dtype=int)
		train_output[-1]=train_output[0]=0

	else:
		train_output = np.array([1,0])
	
	cols = list(map(str,range(1,n+1)))
	train_input = pd.DataFrame(train_input,columns = cols)
	train_output = pd.Series(train_output)
	last = "output"
	return "VALID DATASET",train_input,train_output,train_input,train_output,last

def binary_conversion(num,n):
	binary_str = "{:08b}".format(num)
	binary_num = list(binary_str[8-n:])

	return binary_num

def load_cat_noncat_dataset():
	f1="test_catvnoncat.h5"
	f2="train_catvnoncat.h5"

	with h5py.File(f1,"r") as test:
		X_test=np.array(test["test_set_x"][:])
		Y_test=np.array(test["test_set_y"][:])

	with h5py.File(f2,"r") as train:
		X_train=np.array(train["train_set_x"][:])
		Y_train=np.array(train["train_set_y"][:])

	return "VALID DATASET",X_train,Y_train,X_test,Y_test

def load_internal_csv_dataset(filename,split_percent,algo_type):
	df=pd.read_csv(filename)
	#print(df.head(5))
	cols = list(df.columns)
	last = cols[-1]
	#print(last)
	#randomly shuffle the data which is for test_data
	np.random.seed(2)
	df = df.sample(frac=1).reset_index(drop=True)

	test_data_size = (df.shape[0]*split_percent)//100

	r_no=np.random.randint(df.shape[0]- test_data_size)
	test_data= df[r_no:r_no+ test_data_size]
	train_data=df.drop(range(r_no,r_no+ test_data_size))

	train_output = np.array(train_data[last])
	test_output = np.array(test_data[last])
	
	train_input = train_data.drop([last],axis=1)
	test_input = test_data.drop([last],axis=1)

	return "VALID DATASET",train_input,train_output,test_input,test_output,last

def load_external_csv_dataset(dataset,split_percent,algo_type):
	try:
		df = pd.read_csv(dataset)
		cols = list(df.columns)
		last = cols[-1]
	except:
		return "INVALID FILE FORMAT",None,None,None,None,None

	if df.isnull().values.any():
		return "NAN VALUES IN DATASET",None,None,None,None,None

	#check whether all input data is numeric or not
	for column in cols[:-1]:
		if not is_numeric_dtype(df[column]):
			return "NON NUMERIC VALUE FOUND IN DATASET (EXCLUDING OUTPUT COLUMN)",None,None,None,None,None
	if is_float_dtype(df[last]) and algo_type=="Classification": 
		return "OUTPUT COLUMN DOES NOT CONTAIN CATEGORICAL VALUES",None,None,None,None,None
	if not is_float_dtype(df[last]) and algo_type=="Regression": 
		return "OUTPUT COLUMN DOES NOT CONTAIN CONTINUOUS NUMERIC VALUES",None,None,None,None,None
		
	status="VALID DATASET"
	#randomly shuffle the data first
	np.random.seed(2)
	df = df.sample(frac=1).reset_index(drop=True)
		
	#splitting the training data and testing data
	test_data_size = (df.shape[0]*split_percent)//100
	r_no=np.random.randint(df.shape[0]-test_data_size)
	
	test_data= df[r_no:r_no+test_data_size]
	train_data=df.drop(range(r_no,r_no+test_data_size))
		
	#splitting ouptut and input
	train_output=train_data[last]
	test_output=test_data[last]

	train_input=train_data.drop([last],axis=1)
	test_input=test_data.drop([last],axis=1)

	return status,train_input,train_output,test_input,test_output,last

def load_external_hdf_file(dataset,split):
	
	#File must have only one structure named data
	with h5py.File(dataset,"r") as file:
		df = np.array(file['data'])
	
	test_data_size = (df.shape[0]*split_percent)//100
	r_no=np.random.randint(df.shape[0]-test_data_size)
	test_data= df[r_no:r_no+test_data_size]
	train_data=df.drop(range(r_no,r_no+test_data_size))

	#splitting ouptut and input
	train_output = np.array(train_data[last])
	test_output = np.array(test_data[last])
	train_input = train_data.drop([last],axis=1)
	test_input = test_data.drop([last],axis=1)

	classes = list(train_data[last].unique())

	return train_input,train_output,test_input,test_output,classes,last