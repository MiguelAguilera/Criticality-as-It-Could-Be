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
		
		self.env = gym.make('MountainCar-v0')
		self.env.min_position=-1.5*np.pi/3
		self.env.goal_position=0.5*np.pi/3
		self.env.max_speed=0.045
		self.observation = self.env.reset()
		
		self.Beta=1.0
		self.defaultT=max(100,netsize*20)

		self.Ssize1=int(np.floor(self.Ssize/2))	#size of first sensor
		self.Ssize1=0
		self.maxspeed=self.env.max_speed
		self.sensorbins=np.linspace(-1.01,1.01,2**(self.Ssize-self.Ssize1)+1)
		self.Update(0)
		


	def get_state(self,mode='all'):
		if mode=='all':
			return self.s
		elif mode=='motors':
			return self.s[-self.Msize:]
		elif mode=='sensors':
			return self.s[0:self.Ssize]
		elif mode=='non-sensors':
			return self.s[self.Ssize:]
		elif mode=='hidden':
			return self.s[self.Ssize:-self.Msize]
			
			
	def get_state_index(self,mode='all'):
		return bool2int(0.5*(self.get_state(mode)+1))
			
	#Randomize the state of the network
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1

	def randomize_position(self):
		self.observation = self.env.reset()

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
		self.previous_speed=self.observation[1]
		self.previous_vspeed= self.observation[1]*3*np.cos(3*self.observation[0])
		action=int(np.digitize(np.sum(self.s[-self.Msize:])/self.Msize,[-1/3,1/3,1.1]))
		observation, reward, done, info = self.env.step(action)
		if (self.env.state[0]>=self.env.goal_position and self.env.state[1]>0): 
			self.env.state = (self.env.goal_position,0)  #Bounce when end of world is reached
#		if done:
#			observation = self.env.reset()
		self.observation=self.env.state
		self.position=self.observation[0]
		self.height=np.sin(3*self.position)

		self.speed=self.observation[1]
		self.vspeed=self.speed*3*np.cos(3*self.position)
		self.absspeed=np.sqrt(self.speed**2+self.vspeed**2)
		
		self.acceleration=np.clip((self.speed-self.previous_speed)/0.0035,-1,1)
		self.vacceleration=np.clip((self.vspeed-self.previous_vspeed)/0.0035/3/2,-1,1)

	def UpdateSensors(self):
		self.s[self.Ssize1:self.Ssize]= 2*bitfield(np.digitize(self.speed/self.maxspeed,self.sensorbins)-1,self.Ssize-self.Ssize1)-1

		
	#Execute step of the Glauber algorithm to update the state of one unit
	def GlauberStep(self,i=None):			
		if i is None:
			i = np.random.randint(self.size)
		eDiff = 2*self.s[i]*(self.h[i] + np.dot(self.J[i,:]+self.J[:,i],self.s))
		if self.Beta*eDiff < np.log(1.0/np.random.rand()-1):    # Glauber
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
	def DynamicalCriticalGradient(self,T=None):
		if T==None:
			T=self.defaultT
		
		msH=np.zeros(self.size)
		mF=np.zeros(self.size)
		mG=np.zeros(self.size)
				
		msh=np.zeros(self.size)
		msFh=np.zeros(self.size)
		msGh=np.zeros(self.size)
		mdFh=np.zeros(self.size)
		mdGh=np.zeros(self.size)
		ms2Hh=np.zeros(self.size)
		
		msJ=np.zeros((self.size,self.size))
		msFJ=np.zeros((self.size,self.size))
		msGJ=np.zeros((self.size,self.size))
		mdFJ=np.zeros((self.size,self.size))
		mdGJ=np.zeros((self.size,self.size))
		ms2HJ=np.zeros((self.size,self.size))
		
		# Main simulation loop:
		self.x=np.zeros(T)
		samples=[]
		for t in range(T):
			self.SequentialUpdate()
			self.x[t]=self.position
			H= self.h + np.dot(self.s,self.J)+ np.dot(self.J,self.s)
			F = H*np.tanh(H)-np.log(2*np.cosh(H))
			G = (H/np.cosh(H))**2 + self.s*H*F
			dF = H/np.cosh(H)**2
			dG = 2*H*(1-H*np.tanh(H))/np.cosh(H)**2 + self.s*F + self.s*H*dF
			
			msH+=self.s*H/float(T)
			mF+=F/float(T)
			mG+=G/float(T)
			
			
			msh+=self.s/float(T)
			msFh+=self.s*F/float(T)
			msGh+=self.s*G/float(T)
			mdFh+=dF/float(T)
			mdGh+=dG/float(T)
			ms2Hh+=H/float(T)
			
			for j in range(self.size):
				msJ[j,:]+=self.s*self.s[j]/float(T)
				msFJ[j,:]+=self.s*self.s[j]*F/float(T)
				msGJ[j,:]+=self.s*self.s[j]*G/float(T)
				mdFJ[j,:]+=self.s[j]*dF/float(T)
				mdGJ[j,:]+=self.s[j]*dG/float(T)
				ms2HJ[j,:]+=self.s[j]*H/float(T)
			
		dh = mdGh + msGh - msh*mG - (msh+ms2Hh-msh*msH)*mF - msH*(mdFh+msFh-msh*mF)
		dJ1 = mdGJ + msGJ - msJ*mG - (msJ+ms2HJ-msJ*msH)*mF - msH*(mdFJ+msFJ-msJ*mF)
		
		dJ=np.zeros((self.size,self.size))
		
		dh[0:self.Ssize]=0
		for j in range(self.size):
			for i in range(self.size):
				if i in range(self.Ssize):
					dJ1[j,i]=0				#Remove wrong-way change in unidirectional couplings
					dJ1[i,j]+=dJ1[i,j]		#Multiply by two change in unidirectional couplings
				if i==j:
					dJ1[i,i]=0

		for j in range(self.size):
			for i in range(self.size):
				if i>j:
					dJ[j,i]+=dJ1[j,i]*0.5
				elif j>i:
					dJ[i,j]+=dJ1[j,i]*0.5
		
		self.HCl=mG-msH*mF
		self.HC=np.sum(self.HCl[self.Ssize:])
		
		
		return dh,dJ
		

	#Dynamical Critical Learning Algorithm for poising units in a critical state
	def HomeostaticGradient(self,T=None):
		if T==None:
			T=self.defaultT
		
		self.m=np.zeros(self.size)
		self.c=np.zeros((self.size,self.size))
		self.C=np.zeros((self.size,self.size))
		
		# Main simulation loop:
		self.x=np.zeros(T)
		samples=[]
		for t in range(T):
			
			self.SequentialUpdate()
			self.x[t]=self.position
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
		print(count,fit,np.max(np.abs(self.J)),np.min(self.x+0.5)/np.pi*3,np.max(self.x+0.5)/np.pi*3)
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
			fit=np.mean(np.abs(self.C1[:,self.Ssize:]-self.C[:,self.Ssize:]))
			if count%1==0:
				print(count,fit,np.max(np.abs(self.J)),np.min(self.x[int(T/4):]+0.5)/np.pi*3,np.max(self.x[int(T/4):]+0.5)/np.pi*3)
			

#Transform bool array into positive integer
def bool2int(x):				
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
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

	
