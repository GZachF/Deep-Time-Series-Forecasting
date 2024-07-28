
import os,datetime
import sys
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        scores = self.attention(hidden_states)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * hidden_states, dim=1)
        return context, weights

class AAForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, rnn=True, use_attention=True):
        super(AAForecast, self).__init__()
        self.rnn = rnn
        self.use_attention = use_attention

        if rnn:
            self.rnn_layer = nn.LSTM(input_dim, hidden_dim, num_layers=num_blocks, batch_first=True)
            if use_attention:
                self.attention = Attention(hidden_dim)
        else:
            self.blocks = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_blocks)])
        
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.rnn:
            rnn_out, _ = self.rnn_layer(x)
            if self.use_attention:
                context, _ = self.attention(rnn_out)
                out = self.output_layer(context)
            else:
                out = self.output_layer(rnn_out[:, -1, :])
        else:
            for block in self.blocks:
                x = F.relu(block(x))
            out = self.output_layer(x)
        return out

# Example usage:
input_dim = nv
hidden_dim = HIDDEN
num_blocks = NUMBLOCKS

model = AAForecast(input_dim, hidden_dim, num_blocks, rnn=rnn, use_attention=True if rnn else False)

# Model summary (for illustration purposes)
print(model)


#use rnn? 
rnn=True

#general hyperparameters
BATCHSIZE = 512
NUMBLOCKS = 7
HIDDEN  = 300
SEED=0

#Add extra variables?
AVD=True

#Add additional losses?
IL=True
FIL=True
SL=True
#hyperparams for additional losses
prop=.1*100000
poww=1.0
fpoww=3
proportion=.3

#use weights from a previous run to intialize?
PRETRAIN=True
#Where to look for the previous weights
PTDIR='PRETRAINS/drtf_horizon6/model'

#prediction horizon
horizon=6#

#where to save the outputs (a name for a new folder)
outstr='TEST'


#Do individual subject models
PERSUBJECT=True

subjects=['540','544','552','567','584','596']

loopsthrough=1
if PERSUBJECT:
	loopsthrough=7
	
#NUMBER OF INPUT VARIABLES
nv=1
if AVD:
	nv=7







#All of the backcasts used for ensembling
backcastopts=[2,3,4,5,6,7]




	
#################################### MAIN SECTION ############################################
def main():
	maindir = os.getcwd()+'/'+outstr

	os.makedirs(maindir)
	
	basedir=maindir
	
	#some arrays to store losses for later
	subouts=[]
	submaes=[]
	suboutse=[]
	suboutseh=[]
	suboutsel=[]
	
	#loop through subjects, starting with all subjects
	#(will do all subjects only if PERSUBJECT is false)
	for subb in range(loopsthrough):
		if PERSUBJECT:
			if subb==0:
				sub=99
				maindir=basedir+'/allsubs'
				os.makedirs(maindir)
			else:
				sub=subb-1
				maindir=basedir+'/'+subjects[sub]
				os.makedirs(maindir)
		else:
			sub=99
		curmodel=0
		
		#Training section
		for bc in backcastopts:
			if PRETRAIN and sub==99:
				zerodir=PTDIR+str(bc-2)
			else:
				zerodir=basedir+'/allsubs/model'+str(curmodel)
			print('zerodir:',zerodir)
			np.random.seed(SEED)
			torch.manual_seed(SEED)
			train_and_evaluate(curmodel,maindir,horizon,bc*6,sub,zerodir)
			curmodel=curmodel+1
		
		#final evaluation of ensemble
		#get test data
		train,val,test=makedata(2*6+horizon,sub)
		testgen = ordered_data(BATCHSIZE, 2*6,horizon,test)
		
		#Keep track of total number of evaluated points
		#and total number of each type of event point
		totalpoints=0
		totale=0
		totaleh=0
		totalel=0
		#arrays to store all sorts of losses!
		losses=[]
		rmselosses=[]
		maes=[]
		lossese=[]
		losseseh=[]
		lossesel=[]
		#loop through every batch in training data.
		batch=0
		while(True):
			x,target,done=next(testgen)
			totalpoints = totalpoints+x.shape[0]
			#loop through each directory and load predicions
			preds=[]
			for f in os.listdir(maindir):
				temp=joblib.load(maindir+'/'+f+'/preds.pkl')
				preds.append(temp[batch])
				del temp
			#take median
			preds=np.array(preds)
			median=np.median(preds,axis=0)
			#get losses
			losses.append(mse_cpu(target, median)*x.shape[0])
			rmselosses.append(mse_lastpointonly_cpu(target, median)*x.shape[0])
			maes.append(mae_lastpointonly_cpu(target, median)*x.shape[0])
			
			#event losses- will get MSE of last point
			ee,te=event(target,median,x[:,:,0])
			totale+=te
			lossese.append(ee*te)
			ee,te=eventh(target,median,x[:,:,0])
			totaleh+=te
			losseseh.append(ee*te)
			ee,te=eventl(target,median,x[:,:,0])
			totalel+=te
			lossesel.append(ee*te)
			
			batch=batch+1
			if done:
				break
		
		#write final losses
		#MSE for whole window
		t=open(maindir+"/"+str(np.sum(np.asarray(losses))/totalpoints)+".FINALMSEout","w")
		#Hyper, Hypo, and both evenet RMSEs for last point
		t=open(maindir+"/"+str(np.nansum(np.asarray(lossese))/totale)+".eMSE","w")
		t=open(maindir+"/"+str(np.nansum(np.asarray(losseseh))/totaleh)+".eHMSE","w")
		t=open(maindir+"/"+str(np.nansum(np.asarray(lossesel))/totalel)+".eLMSE","w")
		#number of event windows
		t=open(maindir+"/"+str(totale)+".epts","w")
		t=open(maindir+"/"+str(totaleh)+".eHpts","w")
		t=open(maindir+"/"+str(totalel)+".eLpts","w")
		#rmse and mae for last point only
		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints))+".FINAL_RMSE_out","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(maes))/totalpoints)+".FINAL_MAEout","w")
		
		#collected losses for all subjects, but not all subject run
		if sub!=99:	
			subouts.append(np.sum(np.asarray(rmselosses))/totalpoints)
			submaes.append(np.sum(np.asarray(maes))/totalpoints)
			suboutse.append(np.nansum(np.asarray(lossese))/totale)
			suboutseh.append(np.nansum(np.asarray(losseseh))/totaleh)
			suboutsel.append(np.nansum(np.asarray(lossesel))/totalel)
	if PERSUBJECT:
		#output means across subjects
		subouts=np.array(subouts)
		t=open(basedir+"/"+str(np.mean(np.sqrt(subouts)))+".FINAL_RMSE_out","w")
		subouts=np.array(suboutse)
		t=open(basedir+"/"+str(np.mean(np.sqrt(subouts)))+".eventrmse","w")
		subouts=np.array(suboutseh)
		t=open(basedir+"/"+str(np.mean(np.sqrt(subouts)))+".eventHrmse","w")
		subouts=np.array(suboutsel)
		t=open(basedir+"/"+str(np.mean(np.sqrt(subouts)))+".eventLrmse","w")
		subouts=np.array(submaes)
		t=open(basedir+"/"+str(np.mean(subouts))+".FINAL_MAEout","w")




####################################  TRAINING, AND EVALUATION SECTION ############################################
def train_and_evaluate(curmodel,maindir,forecast_length,backcast_length,sub,basedir):
	mydir = maindir+'/model'+str(curmodel)
	os.makedirs(mydir)
	
	#dump params
	paramlist=[forecast_length,backcast_length]
	joblib.dump(paramlist,mydir+'/params.pkl')
	
	pin_memory=True
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	

	batch_size = BATCHSIZE
	train,val,test=makedata(backcast_length+forecast_length,sub)

	traingen = data(batch_size, backcast_length, forecast_length,train)
	valgen = data(batch_size, backcast_length, forecast_length,val)
	testgen = ordered_data(batch_size, backcast_length, forecast_length,test)
	
	net = network(device,backcast_length,forecast_length,NUMBLOCKS)
	optimiser = optim.Adam(net.parameters(),lr=.0002)

	fit(net, optimiser, traingen,valgen,mydir, device,basedir)
	eval(net, optimiser, testgen,mydir,  device)



def fit(net, optimiser, traingen,valgen,mydir,device, basedir):
	losss=mse_one
	losss2=mse
	losss3=msedoubs
	loadnoopt(net, optimiser,basedir)
	
	
	trains=[]
	vals=[]
	patience=20
	prevvalloss=np.inf
	unimproved=0
		
		
	net.to(device)
	lossbonsum=1
	lossbonsumf=1
	magbonsum=1
	for i in range(NUMBLOCKS):
		lossbonsum = lossbonsum+(i+1)**poww
		if fpoww>0:
			lossbonsumf = lossbonsumf+(i+1)**fpoww
		magbonsum=magbonsum+1/(i+1)
	
	for grad_step in range(500):
		temptrain=[]
		total=0
		while(True):
			optimiser.zero_grad()
			net.train()
			x,target,done=next(traingen)
			total=total+x.shape[0]
			forecast,fores,backs,backsum,backtargs= net(   torch.tensor(x, dtype=torch.float).to(device)	 )
			if FIL:
				loss = 1/lossbonsumf*losss2(fores, torch.tensor(target, dtype=torch.float).to(device))
			else:
				loss = losss(forecast, torch.tensor(target, dtype=torch.float).to(device))
			if AVD:
				x=x[:,:,0]
			if IL:
				loss=loss+proportion/lossbonsum*losss3(backs,backtargs)
			if SL:
				loss=loss+prop/magbonsum*calcsizeloss(backs)
			loss.backward()
			optimiser.step()
			temptrain.append(loss.item()*x.shape[0])
			if done:
				break
		trains.append(np.sum(temptrain)/total)
		print('grad_step = '+str(grad_step)+' loss = '+str(trains[-1]))
		
			
		tempval=[]
		total=0
		while(True):
			with torch.no_grad():
				x,target,done=next(valgen)
				total=total+x.shape[0]
				forecast,fores,backs,backsum,backtargs= net(   torch.tensor(x, dtype=torch.float).to(device)	 )
				loss = losss(forecast, torch.tensor(target, dtype=torch.float).to(device))
				tempval.append(loss.item()*x.shape[0])
				if done:
					break
		vals.append(np.sum(tempval)/total)
		
		print('val loss: '+str(vals[-1]))				

		if vals[-1]<prevvalloss:
			print('loss improved')
			prevvalloss=vals[-1]
			unimproved=0
			save(net, optimiser, grad_step,mydir)
		else:
			unimproved+=1
			print('loss did not improve for '+str(unimproved)+'th time')
		if (unimproved>patience):
			print('Finished.')
			break
	plt.plot(range(len(trains)),trains,'k--', range(len(trains)),vals,'r--')
	plt.legend(['train','val'])
	plt.savefig(mydir+"/loss_over_time.png")
	plt.clf()
	del net
			
			
			
def eval(net, optimiser, testgen,mydir,  device):
	with torch.no_grad():
		load(net, optimiser,mydir)
		totalpoints=0
		losses=[]
		rmselosses=[]
		preds=[]
		while(True):
			x,target,done=next(testgen)
			totalpoints = totalpoints+x.shape[0]
			forecast,dummy1,backs,dummy3,dummy4 = net(torch.tensor(x, dtype=torch.float).to(device))
			preds.append(forecast.cpu().numpy())
			losses.append(mse_one_eval(forecast, torch.tensor(target, dtype=torch.float).to(device)).item()*x.shape[0])
			rmselosses.append(mse_lastpointonly(forecast, torch.tensor(target, dtype=torch.float).to(device)).item()*x.shape[0])
			if done:
				break
		#write final loss
		t=open(mydir+"/"+str(np.sum(np.asarray(losses))/totalpoints)+".testMSEout","w")
		t=open(mydir+"/"+str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints))+".testRmseout","w")
		#dump out predictions to be used in ensembling
		joblib.dump(preds,mydir+'/preds.pkl')

#SAVE AND LOAD FUNCTIONS
def save(model, optimiser, grad_step,mdir):
	torch.save({
		'grad_step': grad_step,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimiser.state_dict(),
	}, mdir+'/model_out.th')


def load(model, optimiser,mdir):
	if os.path.exists(mdir+'/model_out.th'):
		checkpoint = torch.load(mdir+'/'+'model_out.th')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
		grad_step = checkpoint['grad_step']


def loadnoopt(model, optimiser,mdir):
	if os.path.exists(mdir+'/model_out.th'):
		checkpoint = torch.load(mdir+'/'+'model_out.th')
		model.load_state_dict(checkpoint['model_state_dict'])



#LOSS FUNCTIONS
def mse(output, target):
	loss=mse_one(output[0],target)
	for r in np.arange(1,len(output)):
		mul=(r+1)**fpoww
		loss=loss+mul*mse_one(output[r],target)
	return loss  

def msedoubs(output, target):
	loss=mse_one(output[0],target[0])
	for r in np.arange(1,len(output)):
		mul=(r+1)**poww
		loss=loss+mul*mse_one(output[r],target[r])
	return loss  

def mse_one(output, target):
	return torch.mean((output - target)**2)



def calcsizeloss(output):
	for r in np.arange(0,len(output)):
		if r==0:
			loss=1/torch.sum(torch.abs(output[r]))/(r+1)
		else:
			loss=loss+1/torch.sum(torch.abs(output[r]))/(1+r)
	return loss


	


def mse_one_eval(output, target):
	output=output[target!=0]
	target=target[target!=0]
	return torch.mean((output - target)**2)
	

def mse_cpu(output, target):
	output=output[target!=0]
	target=target[target!=0]
	return np.mean((output - target)**2)
	
def mse_lastpointonly(output, target):
	output=output[:,-1]
	target=target[:,-1]
	loss = torch.mean((output - target)**2)
	return loss 

def mse_lastpointonly_cpu(output, target):
	output=output[:,-1]
	target=target[:,-1]
	loss = np.mean((output - target)**2)
	return loss 
	

	
def mae_lastpointonly_cpu(output, target):
	output=output[:,-1]
	target=target[:,-1]
	loss = np.mean(np.abs(output - target))
	return loss 




  
def event(target,output,x):
	output=output[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	target=target[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	eventpoints=(target<70)+(target>180)
	eventwindows=np.max(eventpoints,axis=1)
	output=output[eventwindows,-1]
	target=target[eventwindows,-1]
	return np.mean((output - target)**2),len(target)
	
def eventh(target,output,x):
	output=output[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	target=target[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	eventpoints=(target>180)
	eventwindows=np.max(eventpoints,axis=1)
	output=output[eventwindows,-1]
	target=target[eventwindows,-1]
	return np.mean((output - target)**2),len(target)
	
def eventl(target,output,x):
	output=output[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	target=target[(x[:,-1]<=180)*(x[:,-1]>=70),:]
	eventpoints=(target<70)
	eventwindows=np.max(eventpoints,axis=1)
	output=output[eventwindows,-1]
	target=target[eventwindows,-1]
	return np.mean((output - target)**2),len(target)
	
	
####################################  MODEL SECTION  ############################################################################################################  




class Block(nn.Module):

	def __init__(self, units, device, backcast_length, forecast_length):
		super(Block, self).__init__()
		self.backlen=backcast_length
		self.forecast_length=forecast_length
		self.input=1
		self.outer=1
		if AVD:
			self.input=nv
		self.device = device
		if not rnn:
			self.lin1 = nn.Linear(backcast_length*nv, units)
			self.lin2 = nn.Linear(units, units)
			self.lin3 = nn.Linear(units, units)
			self.lin4 = nn.Linear(units, units)
			self.backcast_layer = nn.Linear(units, units)
			self.forecast_layer = nn.Linear(units, units)
			self.backcast_out = nn.Linear(units, backcast_length)
			self.forecast_out = nn.Linear(units, forecast_length)
		if rnn:
			self.units=HIDDEN#100
			self.bs=BATCHSIZE
			self.lstm=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True)
			self.lin=nn.Linear(self.units *2, (self.backlen+self.forecast_length)*self.outer)
			self.h_0=(torch.zeros(4,self.bs,self.units)).to(self.device)#,
			self.c_0 = (torch.zeros(4,self.bs,self.units)).to(self.device)#,

	def forward(self, x):
		if not rnn:
			if AVD:
				x = F.relu(self.lin1(x.flatten(1,-1).to(self.device)))
			else:	
				x = F.relu(self.lin1(x.to(self.device)))
			x = F.relu(self.lin2(x))
			x = F.relu(self.lin3(x))
			x = F.relu(self.lin4(x))
			theta_b = F.relu(self.backcast_layer(x))
			theta_f = F.relu(self.forecast_layer(x))
			out = self.backcast_out(theta_b) 
			forecast = self.forecast_out(theta_f)
			return out,forecast
		if rnn:
			origbs=x.size()[0]
			if origbs<self.bs:
				if AVD:
					x=F.pad(input=x, pad=( 0,0,0,0,0,self.bs-origbs), mode='constant', value=0)
				else:
					x=F.pad(input=x, pad=( 0,0,0,self.bs-origbs), mode='constant', value=0)
			self.h_0=self.h_0.data
			self.c_0=self.c_0.data
			if not AVD:
				x=x.unsqueeze(2)
			lstm_out, (self.h_0,self.c_0) = self.lstm(x, (self.h_0,self.c_0))
			#print(lstm_out[10,-1,10])
			out=self.lin(lstm_out[:,-1,:])[:origbs,:]
			return out[:,:self.backlen*self.outer],out[:,self.backlen*self.outer:]



class Stack(nn.Module):

	def __init__(self, nb_blocks_per_stack,hidden_layer_units, device,forecast_length,backcast_length):
		super(Stack, self).__init__()
		self.device=device
		self.backcast_length=backcast_length
		self.fl=forecast_length
		self.blocks = nn.ModuleList([Block(hidden_layer_units,device, backcast_length, forecast_length)])
		for block_id in range(1,nb_blocks_per_stack):
			block = Block(hidden_layer_units, device, backcast_length, forecast_length)
			self.blocks.append(block)


		
	def forward(self, x, backcast,backsum,device):
		backs=[]
		fores=[]
		backtargs=[]
		for block_id in range(len(self.blocks)):
			b, f = self.blocks[block_id](backcast)
			#backcast = torch.cat( [(backcast[:,:,0] - b).view([-1,self.backcast_length,1]),backcast[:,:,1:] ],dim=2)
			if AVD:
				backtargs.append(backcast.clone()[:,:,0])
				backcast2=backcast.clone()
				backcast2[:,:,0]=backcast2[:,:,0]-b
				backcast=backcast2
			else:
				backtargs.append(backcast.clone())
				backcast=backcast-b
			backsum=backsum+b
			x= x + f
			#for loss calculation
			fores.append(x.clone())
			backs.append(b.clone())
		return x,backcast,backsum,fores,backs,backtargs





class network(nn.Module):
	def __init__(self,device,backcast_length,forecast_length,nb_stacks):
		super(network, self).__init__()
		self.forecast_length = forecast_length
		self.backcast_length = backcast_length
		self.hidden_layer_units = 512
		self.nb_blocks_per_stack = 1
		self.nb_stacks=nb_stacks
		self.device = device
		self.stacks = nn.ModuleList([Stack( self.nb_blocks_per_stack,self.hidden_layer_units, self.device,forecast_length,backcast_length)])

		for stack_id in range(1,self.nb_stacks):
			self.stacks.append(Stack( self.nb_blocks_per_stack,self.hidden_layer_units, self.device,forecast_length,backcast_length))
		self.to(self.device)



  

	def forward(self, backcast):
		forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,)).to(self.device)  
		backsum = torch.zeros(size=(backcast.size()[0], self.backcast_length,)).to(self.device)  
		fores=[]
		backs=[]
		backtargs=[]
		for stack_id in range(len(self.stacks)):
			forecast,backcast,backsum,curfores,curbacks,cbacktargs =self.stacks[stack_id](forecast,backcast,backsum,self.device)
			for ff in curfores:
				fores.append(ff)
			for ff in curbacks:
				backs.append(ff)
			for ff in cbacktargs:
				backtargs.append(ff)
		return forecast,fores,backs,backsum,backtargs







####################################  DATA GENERATION SECTION  ############################################################################################################  

def makedata(totallength,sub):
	train=[]
	test=[]
	val=[]
	
	stored_trains={}
	#first load train data
	for f in os.listdir('2020data'):
		if f.endswith('train.pkl'):
			if not sub==99: 
				if not f[:3]==subjects[sub]:
					continue
			a=joblib.load('2020data/'+f)
			g=np.asarray(a['glucose'])
			b=np.asarray(a['basal'])
			d=np.asarray(a['dose'])
			c=np.asarray(a['carbs'])
			fing=np.asarray(a['finger'])
			hr=np.asarray(a['hr'])
			gsr=np.asarray(a['gsr'])
			t=np.array(a.index.values)
			t1=np.sin( t*2*np.pi/288)
			t2=np.cos( t*2*np.pi/288)
			miss=(np.isnan(g)).astype(float)
			miss2=(np.isnan(fing)).astype(float)
			x=np.stack((g,d,c,t1,t2,fing,miss),axis=1)
			ll=x.shape[0]
			if not AVD:
				train.append(x.copy()[:int(ll*.8),0])
				val.append(x.copy()[int(ll*.8):,0])
			else:
				train.append(x.copy()[:int(ll*.8),:])
				val.append(x.copy()[int(ll*.8):,:])
			#store to use in test for end
			stored_trains[f]=x.copy()
			
		
			
	for f in os.listdir('2020data'):
		if f.endswith('test.pkl'):
			if not sub==99: 
				if not f[:3]==subjects[sub]:
					continue
			a=joblib.load('2020data/'+f)
			g=np.asarray(a['glucose'])
			b=np.asarray(a['basal'])
			d=np.asarray(a['dose'])
			c=np.asarray(a['carbs'])
			fing=np.asarray(a['finger'])
			hr=np.asarray(a['hr'])
			gsr=np.asarray(a['gsr'])
			t=np.array(a.index.values)
			miss2=(np.isnan(fing)).astype(float)
			t1=np.sin( t*2*np.pi/288)
			t2=np.cos(t *2*np.pi/288)
			miss=(np.isnan(g)).astype(float)
			x=np.stack((g,d,c,t1,t2,fing,miss),axis=1)

			#add in last training points so that we are predicting all points after
			#the first hour of test data
			t=stored_trains[f.replace('test','train')]
			x=np.concatenate((t[-(totallength-12-1):,:],x),axis=0)
			if not AVD:
				test.append(x.copy()[:,0])
			else:
				test.append(x.copy())
	
	return train,val,test




def data(num_samples, backcast_length, forecast_length, data):
		def get_x_y(ii):  
				temp=data[0]
				done=False
				for s in range(len(data)):
						temp=data[s]
						if len(temp)<backcast_length+ forecast_length:
								continue
						if ii<=len(temp)-backcast_length-forecast_length:
								done=True
								break
						ii=ii-(len(temp)-backcast_length-forecast_length)-1
				if not done:
						return None,None,True
								


				i=ii
				learn=temp[i:i+backcast_length]
				see=temp[i+backcast_length:i+backcast_length+forecast_length]
				if AVD:
					see=temp[i+backcast_length:i+backcast_length+forecast_length,0]
				see[np.isnan(see)]=0
				learn[np.isnan(learn)]=0
				if np.prod(see)==0:
					return np.asarray([]),None,False

				return learn,see,False
   
		   
		
		
		def gen():
				done=False
				indices=range(99999999)
				xx = []
				yy = []
				i=0
				added=0
				while(True):
						x, y,done = get_x_y(indices[i])
						i=i+1
						if done or i==len(indices):
								yield np.array(xx), np.array(yy),True
								done=False
								xx = []
								yy = []
								if indices[100]==100 and indices[101]==101:
										indices=np.random.permutation(i-1)
								else:
										indices=np.random.permutation(i)
								i=0
								added=0
								continue
						if not x.shape[0]==0:
								xx.append(x)
								yy.append(y)
								added=added+1
								if added%num_samples==0:
										yield np.array(xx), np.array(yy),done
										xx = []
										yy = []
		return gen()



def ordered_data(num_samples, backcast_length, forecast_length, dataa):
	def get_x_y(i):  
		temp=dataa[0]
		done=False
		for s in range(len(dataa)):
			temp=dataa[s]
			#if this time series is too short, skip it.
			if len(temp)<backcast_length+ forecast_length:
				continue
			#if this index falls within this time series, we can return it
			if i<=len(temp)-backcast_length-forecast_length:
				done=True
				break
			#otherwise subtract this subject's points and keep going.
			i=i-(len(temp)-backcast_length-forecast_length)-1
		#if we're out of data, quit.
		if not done:
			return None,None,True
		learn=temp[i:i+backcast_length]
		see=temp[i+backcast_length:i+backcast_length+forecast_length]
		if AVD:
			see=temp[i+backcast_length:i+backcast_length+forecast_length,0]
		see[np.isnan(see)]=0
		learn[np.isnan(learn)]=0
		#only use data where the point we're trying to predict is there.
		if see[-1]==0:
			return np.asarray([]),None,False
		return learn,see,False
	
	
	
	def gen():
		done=False
		xx = []
		yy = []
		i=0
		added=0
		while(True):
			x, y,done = get_x_y(i)
			i=i+1
			if done:
				yield np.array(xx), np.array(yy),True
				done=False
				xx = []
				yy = []
				i=0
				added=0
				continue
			if not x.shape[0]==0:
				xx.append(x)
				yy.append(y)
				added=added+1
				if added%num_samples==0:
					yield np.array(xx), np.array(yy),False
					xx = []
					yy = []
	return gen()
	


if __name__ == '__main__':
	main()
	
	
	

