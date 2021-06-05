"""Essa biblioteca foi usada para plot"""
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

"""Essas bibliotecas foram usadas só para plotar os gráficos"""
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

"""Essa biblioteca foi usada para utilizar calculos algébricos"""
import numpy.linalg as lin

"""Essa função calcula a distribuição normal multivariada"""
def m_normal(mean,cov,x):
    return (1/(2*np.pi*lin.det(cov)**(1/2)))*np.exp(-(1/2)*np.matmul(np.matmul((x-mean),lin.inv(cov)),np.transpose(x-mean)))
    


"""Essa é a classe em que se emprega o expectation maximization"""
class GMM:
    """simplismente inicializando, com os dados, a decisão do número de gaussianas e interações como input"""
    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        
    
    def run(self):
        """Essa variável é simplismente para evitar que a matriz de cov seja singular"""
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        
        """Essa parte abaixo é referente a plots"""
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
                    
        """ Determina os valores iniciais de mu, cov e pi"""
        self.mu = [[-7,-6],[ 3,-4],[-4,6]] # os mu iniciais foram decididos por meio de testes utilizando pontos aleatórios e escolhendo o que obteve bom resultado
        
        self.cov = np.zeros((self.number_of_sources,len(X[0]),len(X[0]))) 
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],5)
        self.pi = np.ones(self.number_of_sources)/self.number_of_sources
        ########################################
        
        log_likelihoods = [] # Inicializando log de probabilidades
            
        """Plotando o estado inicial"""    
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(self.X[:,0],self.X[:,1])
        ax0.set_title('Initial state')
        for m,c in zip(self.mu,self.cov):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
            ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        
        for i in range(self.iterations):               

            """Expectativa"""
            r_ic = np.zeros((len(self.X),len(self.cov)))

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                for i in range(len(X)):
                    r_ic[i,r] = p*m_normal(m,co,X[i])/np.sum([pi_c*m_normal(mu_c,cov_c,X[i]) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)
            ####################################
            """Maximização"""

            """Calcula os novos mu e cov, baseado nos novos ric(que indica o quao o ponto xi está próximo do cluster c)"""
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                self.mu.append(mu_c)

                # Calcula a matriz de covariância baseada na nova média
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                # calcula o novo pi
                self.pi.append(m_c/np.sum(r_ic))

            
            
            """Log da probabilidade"""
            log_likelihoods.append(np.log(np.sum([k*m_normal(self.mu[i],self.cov[j],X[count]) for k,i,j,count in zip(self.pi,range(len(self.mu)),range(len(self.cov)),range(len(X)))])))

            

        """parte referente a plots"""
        fig2 = plt.figure(figsize=(10,10))
        ax1 = fig2.add_subplot(111) 
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0,self.iterations,1),log_likelihoods)
        #plt.show()
    
    """Parte em que novos pontos para o dataset são classificados"""
    def predict(self,Y):
        # Plota o ponto nas gaussianas que foram ajustadas
        """Toda essa parte diz respeito ao plot"""
        fig3 = plt.figure(figsize=(10,10))
        ax2 = fig3.add_subplot(111)
        ax2.scatter(self.X[:,0],self.X[:,1])
        for m,c in zip(self.mu,self.cov):
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax2.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
            ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            ax2.set_title('Final state')
            for y in Y:
                ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
        """Até aqui foi com respeito ao plot"""
        
        prediction = []        
        for m,c in zip(self.mu,self.cov):
            """"A CLASSIFICAÇÃO É DETERMINADA AQUI!"""
            prediction.append(m_normal(m,c,Y)/np.sum([m_normal(mean,cov,Y) for mean,cov in zip(self.mu,self.cov)]))
        return prediction
         
"""O código abaixo pode ser alterado para se obter novos exemplos"""

# Criando um dataset
X,Y = make_blobs(cluster_std=1.5,random_state=20,n_samples=500,centers=3)

    
gaussian_mix = GMM(X,3,40)     
gaussian_mix.run()

"""Temos exemplos de classificação abaixo"""

print("Para o ponto [-10,2] temos:")
print(gaussian_mix.predict([[-10,2]]))
print("ou seja ele é classificado como [1 0 0]\n\n")

print("Para o ponto [10,6] temos:")
print(gaussian_mix.predict([[10,6]]))
print("ou seja ele é classificado como [0 1 0]\n\n")

print("Para o ponto [0,8] temos:")
print(gaussian_mix.predict([[0,8]]))
print("ou seja ele é classificado como [0 0 1]")
