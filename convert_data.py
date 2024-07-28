import xml.etree.ElementTree as etree
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt


#call this code from the directory that contains the data

outdir='data2020'
os.makedirs('../'+outdir)

columns = ['glucose', 'finger', 'basal', 'hr', 'gsr','carbs','temp_basal', 
	'dose','bwz_carb_input' ]
       
xmlkeys=["glucose_level","finger_stick","basal","basis_heart_rate","basis_gsr",
	"meal","temp_basal","bolus",]
	
dict={}


for fff in os.listdir('.'):
	if not fff.endswith('.xml'):
		continue
	tree = etree.parse(fff)
	finaltime=[]
	#loop through outpus
	num=0
	for x in xmlkeys:
		time=[]
		val=[]
		val2=[]
		time2=[]
		rawtime=[]
		#loop through instancies
		for f in tree.iter(x):
			#actual instances loop

			for g in f:
				#divide time by 300 to get 5 minute intervals
				if num<5:
					val.append(float(g.items()[1][1]))
					time.append(pd.to_datetime(g.items()[0][1],dayfirst=True).timestamp()/300)
					rawtime.append(g.items()[0][1])
				if num==5:
					val.append(float(g.items()[2][1]))
					time.append(pd.to_datetime(g.items()[0][1],dayfirst=True).timestamp()/300)
				if num==6:
					val.append(float(g.items()[2][1]))
					time.append(pd.to_datetime(g.items()[0][1],dayfirst=True).timestamp()/300)
					time2.append(pd.to_datetime(g.items()[1][1],dayfirst=True).timestamp()/300)
				if num==7:
					val.append(float(g.items()[3][1]))
					time.append(pd.to_datetime(g.items()[0][1],dayfirst=True).timestamp()/300)
					time2.append(pd.to_datetime(g.items()[1][1],dayfirst=True).timestamp()/300)
		if len(time)==0:
			if num==6:
				num=num+1
				continue;
		time=np.array(time)
		val=np.array(val)
		sorter=np.argsort(time)
		time=time[sorter]
		val=val[sorter]

		if num>5:
			time2=np.array(time2)
			time2=time2[sorter]
		
		#get basetime	
		if num==0:
			if 'test' in fff:
				joblib.dump(rawtime,'../'+outdir+'/'+fff[:3]+'.timestamps.pkl')
			basetime=np.linspace(time.copy()[0],time.copy()[-1]+1,time.copy()[-1]+1-time.copy()[0])#-time.copy()[0]
			dict['']=basetime
			zerotime=time.copy()[0]
			out=np.array(val)
		#do interpolation
		time=np.array(time)-zerotime
		val=np.array(val)
		out=np.full(len(basetime),np.nan)
		#for basal and basal 0s, use carry forward imputation
		if num==2:
			for i in range(len(time)):
				if int(time[i])< len(basetime):
					out[int(time[i]):]=val[i]
		#basal 0s just shows when the pump is off so update basal array
		elif num==6:
			out=dict['basal']
			time2=np.array(time2)-zerotime
			for i in range(len(time)):
				if int(time[i])< len(basetime):
					out[int(time[i]):int(time2[i])]=val[i]
		#For other variables, just put each value at the closest 5 minute time point.
		else:	
			for i in range(len(time)):
				if int(time[i])< len(basetime):	
					out[int(time[i])]=val[i]

		#add to dictionary
		if num==6:
			dict['basal']=out
		else:
			dict[columns[num]]=out
		#move onto next.	
		num=num+1
	
	#save data frame
	df=pd.DataFrame(dict)
	df.set_index('')
	if 'test' in fff:
		joblib.dump(df,'../'+outdir+'/'+fff[:3]+'.test.pkl')
	if 'train' in fff:
		joblib.dump(df,'../'+outdir+'/'+fff[:3]+'.train.pkl')
	