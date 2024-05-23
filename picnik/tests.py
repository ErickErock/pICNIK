import unittest
import picnik as pic
from rxn_models import rxn_models
import pandas as pd
import numpy as np



class compensation_effect_tests(unittest.TestCase):

	def test_load_data(self):
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		alpha = pd.read_csv("alpha.csv")
		temp = pd.read_csv("temp.csv")
		temp = temp.set_index(alpha['alpha'].values)
		diff = pd.read_csv("diff.csv")
		diff = diff.set_index(alpha['alpha'].values)
		ae = pic.ActivationEnergy(Beta, T0, TempIsoDF=temp, diffIsoDF=diff)
		def const(a, integral = False):
			return 1
		
		E = pd.read_csv("E.csv")
		B = 10.
		f_alpha = [const]
		col = temp.columns[2]
		rightX = temp.loc[(temp.index > 0.005) & (temp.index < 0.995)][col].values
		rightY = diff.loc[(diff.index > 0.005) & (diff.index < 0.995)][col].values
		alpha = temp.loc[(temp.index > 0.005) & (temp.index < 0.995)].index.values
		rightfAlpha = f_alpha
		rightfAlpha += filter(callable, list(rxn_models.__dict__.values()))
		x,y,falpha,alphav = ae._load_data(B,f_alpha)

		self.assertTrue(np.all(x == rightX),'wrong x output values')
		self.assertTrue(np.all(y == rightY),'wrong y output values')
		self.assertTrue(np.all(falpha == rightfAlpha),'wrong alpha functions output values')
		self.assertTrue(np.all(alphav == alpha),'wrong alpha output values')

	def test_fitting(self):
		rng = np.random.default_rng()

		f = []
		f += filter(callable, list(rxn_models.__dict__.values()))

		x = np.linspace(300,600)
		alpha = np.random.uniform(0.1,0.95)
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		ae = pic.ActivationEnergy(Beta, T0)
		for f_i in f:
			A = np.exp(np.random.uniform(2,7))
			E = np.random.uniform(2,7)
			y = A*np.exp(-E/(0.0083144626*x))*f_i(alpha)
			noiseMax = (np.mean(y)) * 0.1
			noiseMin = - noiseMax
			yNoise = rng.uniform(noiseMin,noiseMax,size=y.size)
			y += yNoise
			a,e=ae._fit(x,y,[f_i],alpha)
			self.assertTrue(abs(A-a[0])/A<=0.2,'prexponential factor value not fitted correctly')
			self.assertTrue(abs(E-e[0])/E<=0.2,'activation energy value not fitted correctly')

	def test_regression(self):
		rng = np.random.default_rng()

		def line(x,a,b):
			return b+(a*x)

		A = np.random.uniform(2,7)
		B = np.random.uniform(2,7)

		x = np.linspace(0.0, 1, 100)
		y = line(x,A,B)
		noiseMax = np.mean(y) * 0.1
		noiseMin = -noiseMax
		yNoise = rng.uniform(noiseMin,noiseMax,size=y.size)
		y += yNoise

		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		ae = pic.ActivationEnergy(Beta, T0)
		a,b =ae._regression(x,np.exp(y))
		
		self.assertTrue(abs(A-a)/A<=0.2, 'slope value is too far off of real value')
		self.assertTrue(abs(B-b)/B<=0.2, 'intercept value is too far off of real value')

class reconstruction_tests(unittest.TestCase):

	def test_reconstruction(self):
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		time = pd.read_csv("timeAdvIsoDF.csv",index_col=0)
		ae = pic.ActivationEnergy(Beta, T0, timeAdvIsoDF=time)
		E = 75*(np.ones(len(time)))
		A = 12*(np.ones(len(time)))
		index = np.random.randint(len(Beta))
		b = Beta[index]
		
		g = ae.reconstruction(E,np.exp(A),b)

		self.assertTrue(np.mean(g)-1<0.2, 'reconstruction value is too far of of real value')

class prediction_modelfree_tests(unittest.TestCase):

	def test_modelfree_isotermic_case(self):
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		time = pd.read_csv("timeAdvIsoDF.csv",index_col=0)
		ae = pic.ActivationEnergy(Beta, T0, timeAdvIsoDF=time)

		E = 75*(np.ones(len(time)))
		T_init = T0[0] + np.random.uniform(-1,1)
		alpha = np.random.uniform(0.1,0.95)
		bounds = (np.random.uniform(7,20),np.random.uniform(7,20))
		isoT = np.random.uniform(450,600)
		B = 0

		a_prime, T_prime, t_prime = ae.modelfree_prediction(E, B, isoT=isoT,
		 T_init=T_init, alpha=alpha, bounds = bounds)

		def alpha_F1_iso(t,A,E,isoT):
			return 1- np.exp(-A*np.exp(-E/(0.0083144626*isoT))*t)

		time_teo = np.linspace(0,t_prime[-1],300)
		alpha_teo  = alpha_F1_iso(time_teo,np.exp(12),75,isoT)

		def diffProm(a,b):
			return abs(np.mean(a)-np.mean(b))/np.mean(a)

		self.assertTrue(diffProm(time_teo,t_prime)<=0.5, 'time prediction values are too off of actual values')
		self.assertTrue(np.all(T_prime == isoT), 'temperatue prediction values are too off of actual values')
		self.assertTrue(diffProm(a_prime,alpha_teo)<=0.5, 'conversion prediction values are too off of actual values')

	def test_modelfree_base_case(self):
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([301.15, 302.15, 303.15, 304.15])
		time = pd.read_csv("timeAdvIsoDF.csv",index_col=0)
		ae = pic.ActivationEnergy(Beta, T0, timeAdvIsoDF=time)

		E = 75*(np.ones(len(time)))
		c = np.random.randint(0,3)
		B = Beta[c]
		col = time.columns[c]

		alpha = np.random.uniform(0,0.95)
		bounds = (np.random.uniform(7,20),np.random.uniform(7,20))
		T_init = T0[c] + np.random.uniform(-1,1)
		
		a_prime, T_prime, t_prime = ae.modelfree_prediction(E, B,
		 T_init=T_init, alpha=alpha, bounds = bounds)

		if alpha != 0:
			time_filtered = time.loc[time.index<=alpha]
		else:
			time_filtered = time
		alpha_teo = time_filtered.index.values
		time_teo = time_filtered[col].values
		temp_teo = T_init + (B*time_teo)

		def diffProm(a,b):
			return abs(np.mean(a)-np.mean(b))/np.mean(a)

		self.assertTrue(diffProm(time_teo,t_prime)<=0.5, 'time prediction values are too off of actual values')
		self.assertTrue(diffProm(temp_teo,T_prime)<=0.5, 'temperatue prediction values are too off of actual values')
		self.assertTrue(np.all(a_prime==alpha_teo), 'conversion prediction values aren\'t equal to the actual values')

	def test_modelfree_custom_function_case(self):
		Beta = np.array([ 2.5,  5. , 10. , 20. ])
		T0 = np.array([303.15, 303.15, 303.15, 303.15])
		time = pd.read_csv("timeAdvIsoDF.csv",index_col=0)
		ae = pic.ActivationEnergy(Beta, T0, timeAdvIsoDF=time)

		E = pd.read_csv("E.csv")["Ea"].values
		T_init = T0[0] + np.random.uniform(-1,1)
		alpha = np.random.uniform(0.1,0.95)
		bounds = (np.random.uniform(7,20),np.random.uniform(7,20))
		B = 0
		

		def Temp_program(t):
			T = []
			for i in t:
				if i > 25:
					T += [575]
				else:
					T += [450 + 5*i]
			return np.array(T)

		a_prime, T_prime, t_prime = ae.modelfree_prediction(E, B, T_func=Temp_program,
		 T_init=T_init, alpha=alpha, bounds = bounds)

		def alp_val(t):
			J = []
			A = np.exp(12)
			sumJ = 0
			for i in range(0,len(t)-1):
				sumJ += ae.J_time(75,0,0,ti = t[i], tf=t[i+1], T_func=Temp_program)
				J += [sumJ] 
			J = np.array(J)
			return 1 - np.exp(-A*J)

		time_teo = np.linspace(0,t_prime[-1],300)
		alp_teo  = alp_val(time_teo)
		temp_teo = Temp_program(time_teo)

		def diffProm(a,b):
			return abs(np.mean(a)-np.mean(b))/np.mean(a)
		
		self.assertTrue(diffProm(time_teo,t_prime)<=0.5, 'time prediction values are too off of actual values')
		self.assertTrue(diffProm(temp_teo,T_prime)<=0.5, 'temperatue prediction values are too off of actual values')
		self.assertTrue(diffProm(a_prime,alp_teo)<=0.5, 'conversion prediction values are too off of actual values')
		


if __name__ == '__main__':
	unittest.main()