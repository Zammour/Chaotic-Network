# Chaotic-Network


g_GG=1.5;
g_Gz=1;
g_GF=0;


N_G=1000;
p_GG=0.1;
p_Z=1;

np.random.seed(1);

<!---what is the dimension of w? then figure out z--->

w=np.array(N_G,N_G);
w_nonzero=np.random.normal(0,1/(p_Z*N_G),N_G-1);
for i in range(N_G):
   
    if i<1: 
        w[i]=0;
    else:
        w[i]=w_nonzero[i];
random.shuffle(w);

J_GG=np.random.normal(0,1/(p_Z*N_G),(N_G,N_G));
J_Gz=np.random.uniform(-1,1,(N_G,N_z));


tau=10;

X0=0.1

<!---what is the total time?--->
def g_GG_simul(N_G=N_G,dt=tau,T=3000):
    xt=np.zeros(T);
    xt[0]=x0;
    for i in range(1,T):
        for j in range(1,T):
            xt[i]=xt[i-1]+(-xt[i-1]+g_GG*(J_GG*np.tanh(xt[i-1])))+g_Gz*J_Gz*z
   <!---return xt,dt*np.linspace(0,r,r);--->
   
   <!---do we use both gf and gg networks? --->
