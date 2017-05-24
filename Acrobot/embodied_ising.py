import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import gym


class ising:
	#Initialize the network
	def __init__(self, netsize,Nsensors=1,Nmotors=1):	#Create ising model
	
		self.size=netsize
		self.Ssize=Nsensors			#Number of sensors
		self.Msize=Nmotors			#Number of sensors
		
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.max_weights=2
		self.randomize_state()
		
		self.env = gym.make('Acrobot-v1')
		mass=1.75
		self.set_mass(mass)
		self.observation = self.env.reset()
		self.randomize_position()
		
		self.Beta=1.0
		self.defaultT=max(100,netsize*20)
		self.maxspeed=self.env.MAX_VEL_1
		self.minspeed=-self.env.MAX_VEL_1
		self.maxspeed2=self.env.MAX_VEL_2
		self.minspeed2=-self.env.MAX_VEL_2
		self.maxacc=5
		self.minacc=-5
		self.maxheight=2
		self.minheight=-2
		self.Ssize1=int(np.floor(self.Ssize/2))
		self.sensorbins=np.linspace(-1.01,1.01,2**(self.Ssize1)+1)
			
		self.Update(0)
		
		
	def get_state(self,mode='all'):
		if mode=='all':
			return self.s
		elif mode=='motors':
			return self.s[-self.Msize:]
		elif mode=='sensors':
			return self.s[0:self.Ssize]
	
	def get_state_index(self,mode='all'):
		return bool2int(0.5*(self.get_state(mode)+1))
			
	def set_mass(self,mass=1):
		self.env.LINK_MASS_1=mass
		self.env.LINK_MASS_2=mass
	#Randomize the state of the network
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1

	def randomize_position(self):
		self.observation = self.env.reset()
		self.theta1_dot=0
		
	#Set random bias to sets of units of the system
	def random_fields(self,max_weights=None):
		if max_weights is None:
			max_weights=self.max_weights
		self.h[self.Ssize:]=max_weights*(np.random.rand(self.size-self.Ssize)*2-1)		
		
	#Set random connections to sets of units of the system
	def random_wiring(self,max_weights=None):	#Set random values for h and J
		if max_weights is None:
			max_weights=self.max_weights
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				if i<j and (i>=self.Ssize or j>=self.Ssize):
					self.J[i,j]=(np.random.rand(1)*2-1)*self.max_weights
		
	def Move(self):
		action=int(np.digitize(np.sum(self.s[-self.Msize:])/self.Msize,[-1/3,1/3,1.1]))
		observation, reward, done, info = self.env.step(action)
		theta1_dot_p=self.theta1_dot
		self.theta1_dot=self.env.state[2]
		self.theta1_dot2=self.theta1_dot-theta1_dot_p
		
	def SensorIndex(self,x,xmax):
		return np.digitize(np.clip(x,-xmax,xmax)/xmax,self.sensorbins)-1
		
	def UpdateSensors(self):
		self.acc=self.theta1_dot2
		s = self.env.state
		self.theta=s[0]
		
		self.ypos0=-np.cos(s[0])
		self.xpos0=np.sin(s[0])
		
		self.ypos=-np.cos(s[0]) - np.cos(s[1] + s[0])
		self.xpos=np.sin(s[0]) + np.sin(s[1] + s[0])
		
		self.speed=self.env.state[2]
		self.speedx=self.speed*np.cos(s[0])
		self.speedy=self.speed*np.sin(s[0])
		
		self.speed2=self.env.state[3]
		
		self.posx0_ind=self.SensorIndex(self.xpos,self.maxheight/2)
		self.posy0_ind=self.SensorIndex(self.ypos,self.maxheight/2)
		
		self.posx_ind=self.SensorIndex(self.xpos,self.maxheight)
		self.posy_ind=self.SensorIndex(self.ypos,self.maxheight)
		
		self.speed_ind=self.SensorIndex(self.speed,self.maxspeed)
		self.speed2_ind=self.SensorIndex(self.speed2,self.maxspeed2)
		self.speedx_ind=self.SensorIndex(self.speedx,self.maxspeed)
		self.speedy_ind=self.SensorIndex(self.speedy,self.maxspeed)
		self.speed2_ind=self.SensorIndex(self.speed2,self.maxspeed2)
		
		self.acc_ind=self.SensorIndex(self.acc,self.maxacc)
		self.accx_ind=self.SensorIndex(self.acc*np.cos(s[0]),self.maxacc)
		self.accy_ind=self.SensorIndex(self.acc*np.sin(s[0]),self.maxacc)
		
		self.s[self.Ssize1:self.Ssize]= 2*bitfield(self.posx_ind,self.Ssize-self.Ssize1)-1
		self.s[0:self.Ssize1]=2*bitfield(self.posy_ind,self.Ssize1)-1
#		
		
	#Execute step of the Glauber algorithm to update the state of one unit
	def GlauberStep(self,i=None):			
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
			self.s[i] = -self.s[i]
			
	#Compute energy difference between two states with a flip of spin i	
	def deltaE(self,i):		
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
		
		
	#Update states of the agent from its sensors		
	def Update(self,i=None):	
		if i is None:
			i = np.random.randint(self.size)
		if i==0:
			self.Move()
			self.UpdateSensors()
		elif i>=self.Ssize:
			self.GlauberStep(i)
	
	#Update states of the agent in dreaming mode			
	def UpdateDreaming(self,i=None):	
		if i is None:
			i = np.random.randint(self.size)
		self.GlauberStep(i)
			
	def SequentialUpdate(self):
		for i in np.random.permutation(self.size):
			self.Update(i)

	def SequentialUpdateDreaming(self):
		for i in np.random.permutation(self.size):
			self.UpdateDreaming(i)
	
	#Update all states of the system without restricted infuences
	def SequentialGlauberStep(self):	
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)

	#Dynamical Critical Learning Algorithm for poising units in a critical state
	def HomeostaticGradient(self,T=None):
		if T==None:
			T=self.defaultT
		
		self.m=np.zeros(self.size)
		self.c=np.zeros((self.size,self.size))
		self.C=np.zeros((self.size,self.size))
		
		# Main simulation loop:
		self.y=np.zeros(T)
		samples=[]
		for t in range(T):
			
			self.SequentialUpdate()
			self.y[t]=self.ypos
			self.m+=self.s
			for i in range(self.size):
				self.c[i,i+1:]+=self.s[i]*self.s[i+1:]
		self.m/=T
		self.c/=T
		for i in range(self.size):
			self.C[i,i+1:]=self.c[i,i+1:]-self.m[i]*self.m[i+1:]		
			
		dh = self.m1-self.m
		dJ = self.C1-self.C
		dh[0:self.Ssize]=0
		dJ[0:self.Ssize,0:self.Ssize]=0
		dJ[-self.Msize:,-self.Msize:]=0
		dJ[0:self.Ssize,-self.Msize:]=0
		return dh,dJ
		
		
	def CriticalLearning(self,Iterations,T=None):	
		u=0.01
		count=0
		dh,dJ=self.HomeostaticGradient(T)
		fit=max(np.max(np.abs(self.C1-self.C)),np.max(np.abs(self.m1-self.m)))
		print(count,fit,self.env.LINK_MASS_1,np.max(np.abs(self.J)),np.mean(self.y),np.max(self.y))
		self.l2=0.004
		for i in range(Iterations):
			count+=1
			self.h+=u*dh - self.l2*self.h
			self.J+=u*dJ - self.l2*self.J
			
			Vmax=self.max_weights
			for i in range(self.size):
				if np.abs(self.h[i])>Vmax:
					self.h[i]=Vmax*np.sign(self.h[i])
				for j in np.arange(i+1,self.size):
					if np.abs(self.J[i,j])>Vmax:
						self.J[i,j]=Vmax*np.sign(self.J[i,j])
						
			dh,dJ=self.HomeostaticGradient(T)
			fit=np.mean(np.abs(self.C1[:-self.Msize,self.Ssize:]-self.C[:-self.Msize,self.Ssize:]))
		
			
			if count%1==0:
				print(count,fit,self.env.LINK_MASS_1,np.max(np.abs(self.J)),np.mean(self.y),np.max(self.y))
			

#Transform bool array into positive integer
def bool2int(x):				
	y = 0
	for i,j in enumerate(np.array(x)[::-1]):
#		y += j<<i
		y += j*2**i
	return int(y)

#Transform positive integer into bit array
def bitfield(n,size):
	x = [int(x) for x in bin(int(n))[2:]]
	x = [0]*(size-len(x)) + x
	return np.array(x)

#Extract subset of a probability distribution
def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
#Compute Entropy of a distribution
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log(P[n])
	return E
	
#Compute Mutual Information between two distributions
def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
#Compute TSE complexity of a distribution
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
#Compute the Kullback-Leibler divergence between two distributions
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
 
#Compute the Jensen-Shannon divergence between two distributions   
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))

	
