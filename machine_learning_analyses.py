# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.lines as mlines

# UNIVERSAL CONSTANTS
F=96485
FRT = 96485/8.314/300
ABC = ["A", "B","C","D","E"]
#
plt.rcParams["font.size"] = 20 
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.major.size"] =8.0
plt.rcParams["ytick.major.size"] = 8.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0
plt.rc('legend', fontsize=14)
plt.rcParams['lines.markersize'] =3
AREA = np.pi * 0.55*0.55/4
MECHANISM = "2"
E1_LOCK = 0 # Set it to 0 to unlock E1 
E2_LOCK = 0 # Set it to 0 to unlock E2 
K10_LOCK = 0 # Set it to 0 to unlock K10
k20_LOCK = 0 # Set it to 0 to unlock k20
ALPHA_LOCK = 0.5 # Set it to 0 to unlock alpha 
aH2O = 1000/18
print("aOH-を入力してください")
aOH = float(input())

# %%
#input parameterを設定し電流のlogを返す
def get_log_j(par_vec):
    """
    The par_vec to be inputted:
    E1,E2,logK10,logk20,a = par_vec
    a: Electron transfer coefficient of first step
    """
    E1,E2,logK10,logk20,a = par_vec
    K10 = 10**logK10
    k20 = 10**logk20
    if E1_LOCK != 0:
        E1 = E1_LOCK
    if E2_LOCK != 0:
        E2 = E2_LOCK
    if K10_LOCK != 0:
        K10 = K10_LOCK
    if k20_LOCK != 0:
        k20 = k20_LOCK
    if ALPHA_LOCK != 0:
        a = ALPHA_LOCK
    j = F*K10*k20*aOH**2*np.exp(FRT*(E-E1))*np.exp((1-a)*FRT*(E-E2))/(1+K10*aOH*np.exp(FRT*(E-E1)))
    return np.log10(np.abs(j))

#電流の理論値と実測値の差の平均(offset)を返す   
def get_offset(log_j):
    offset = np.average(log_j - log_j_exp) # treat log_j_exp as a global parameter
    return offset

#電流の理論値とoffsetの差を返す
def get_offset_log_j(log_j):
    o = log_j - get_offset(log_j)
    return log_j - get_offset(log_j)

#電流の理論値とoffsetの差-実測値の電流の2乗の平均(R^2)を返す
def get_diff(par_vec):
    log_j = get_log_j(par_vec)
    diff = np.average((get_offset_log_j(log_j) - log_j_exp)**2)
    return diff

#R^2のlogを返す
def get_log_diff(par_vec):
    log_j = get_log_j(par_vec)
    o = (get_offset_log_j(log_j) - log_j_exp)**2
    diff = np.average((get_offset_log_j(log_j) - log_j_exp)**2)
    log_diff = np.log10(diff)
    return log_diff

# %%
#遺伝的アルゴリズム：交叉
#ある重みで親をランダムに変換して子を返す
def crossover(parents):
    children = np.ones(shape = parents.shape)
    for idx in range(parents.shape[1]-1):
        children[:,idx] = random.choices(parents[:,idx], weights = 1/10**parents[:,-1], k = parents.shape[0])
    if E1_LOCK != 0:
        children[:,-6] = E1_LOCK
    if E2_LOCK != 0:
        children[:,-5] = E2_LOCK
    if K10_LOCK != 0:
        children[:,-4] = K10_LOCK
    if k20_LOCK != 0:
        children[:,-3] = k20_LOCK
    if ALPHA_LOCK != 0:
        children[:,-2] = ALPHA_LOCK    
    return children

def evolve(parents):
    #ユニークな行の抽出とそれぞれの値の頻度
    unique, count = np.unique(parents, axis=0, return_counts=True)
    #頻度が1より大きければ重複
    duplicates = unique[count > 1]
    for duplicate in duplicates:
        #重複しているインデックス
        repeated_idx = np.argwhere(np.all(parents == duplicate, axis = 1))
        #重複しているインデックス数をevol_timeとする
        evol_time = len(repeated_idx)
        evolved_parent = GD(duplicate, evol_time)
        parents[repeated_idx[0,0]] = evolved_parent
    return parents   

#deltaだけ動かしたときのR^2を比較して挿入
def GD(parent, evol_time):
    delta = 1E-5
    GD_cycles = 10 * evol_time
    old_par_vec = parent[0:-1] # actually has log_diff at the end
    old_diff = get_diff(old_par_vec)
    grad_vec = np.zeros(num_pars)

    for cycles in range(GD_cycles):
        for idx in range(num_pars):
            if E1_LOCK != 0:
                if idx == 0:
                    continue
            if E2_LOCK != 0:
                if idx == 1:
                    continue
            if K10_LOCK != 0:
                if idx == 2:
                    continue
            if k20_LOCK != 0:
                if idx == 3:
                    continue
            if ALPHA_LOCK != 0:
                if idx == 4:
                    continue
            delta_vec = np.zeros(num_pars)
            delta_vec[idx] = delta
            # print(delta_vec[idx])
            grad_vec[idx] =(get_diff(old_par_vec+delta_vec) - old_diff)/delta

        new_par_vec = old_par_vec - 0.01 * grad_vec
        new_diff = get_diff(new_par_vec)
        if new_diff < old_diff:
            old_par_vec = new_par_vec
            old_diff = new_diff
        else:
            delta = delta/10
    old_log_diff = np.log10(old_diff)
    return np.append(old_par_vec, old_log_diff)
    
#%%##########################################
############ IMPORT DATA ####################
#############################################
filename = "OER.csv"
df = pd.read_csv(filename)
all_E = df.dropna()["Ecorr_3600"] # units: V vs. RHE after IR correction
all_j_exp = df.dropna()["i_3600"]/AREA # units: mA/cm2 geometric area of RDE 

j_exp = all_j_exp[10**-3<all_j_exp].values # choose region for fitting
E = all_E[10**-3<all_j_exp].values # choose region for fitting
all_E = all_E.values
all_j_exp = all_j_exp.values
log_j_exp = np.log10(np.abs(j_exp))
all_log_j_exp = np.log10(np.abs(all_j_exp))

# %%
pop_size = 1000
num_generations = 50
num_parents = int(pop_size * 0.5)
num_children = pop_size - num_parents 
num_pars = 5 # E1,E2,logK10,logk20,a
num_trials = 100
results_table = np.zeros((num_trials,num_pars+1))
# %%
for trial in range(num_trials):
    print(trial)
    pop = np.zeros((pop_size,num_pars + 1)) #pars +  diff
    pop[:,0] = np.random.uniform(1, 2, pop_size) # 1 <= E1 < 2
    pop[:,1] = np.random.uniform(1, 2, pop_size) # 1 <= E2 < 2
    pop[:,2] = np.random.uniform(-10, 10, pop_size) # -10 <= logK10 < 10
    pop[:,3] = np.random.uniform(-10, 10, pop_size) # -10 <= logk20 < 10
    pop[:,4] = np.random.uniform(0, 1, pop_size) #0 <= alpha < 1
    if E1_LOCK != 0: # Set E_LOCK to zero if E1 should be a free variable
        pop[:,0] = E1_LOCK    
    if E2_LOCK != 0: # Set E_LOCK to zero if E2 should be a free variable
        pop[:,1] = E2_LOCK    
    if K10_LOCK != 0: # Set K10_LOCK to zero if K10 should be a free variable
        pop[:,2] = K10_LOCK
    if k20_LOCK != 0: # Set k20_LOCK to zero if k20 should be a free variable
        pop[:,3] = k20_LOCK
    if ALPHA_LOCK !=0: # Set ALPHA_LOCK to zero if alpha should be a free variable
        pop[:,4] = ALPHA_LOCK
    for p in range(pop_size):
        pop[p,-1] = get_log_diff(pop[p,0:-1])
    pop = pop[np.argsort(pop[:,-1])]
    pop_init = pop
    pop_dict= np.zeros((num_generations,pop_size,num_pars+1))
    for g in range(num_generations):
        pop_dict[g] = pop    
        parents = pop[0:num_parents]
        children = crossover(parents)
        evolved_parents = evolve(parents)
        pop = np.vstack((evolved_parents,children))
        for p in range(pop_size):
            pop[p,-1] = get_log_diff(pop[p,0:-1])
        pop = pop[np.argsort(pop[:,-1])]
        if pop[0,-1] == pop[-1,-1]:
            print("Optimization Converged at Generation : " + str(g))
            break
        plt.plot(pop[:,-1])
    results_table[trial] = pop[0]
# %%    
np.savetxt("results_table_" + MECHANISM + "_" + str(pop_size) + ".csv", results_table, delimiter=',', header="E1,E2,logK10,logk20,a,log diff",comments='' )    
np.savetxt("pop_dict_" + MECHANISM + "_" + str(pop_size) + ".csv", pop_dict.reshape(pop_size*num_generations,num_pars+1), delimiter=',', header="E1,E2,logK10,logk20,a,log diff",comments='' )    
#%%# STATISTICS

#pop_list = [10,30,50,100,300,500,1000]
pop_list = [1000]
M_Mean = np.zeros((len(pop_list), num_pars+2))
M_STD =  np.zeros((len(pop_list), num_pars+2))

for idx, pop in enumerate(pop_list):
    filename = ("results_table_MECHANISM_pop.csv").replace('MECHANISM', MECHANISM).replace('pop', str(pop))
    df = pd.read_csv(filename)
    M_Mean[idx,1:] = df.mean()
    M_STD[idx,1:] = df.std()
M_Mean[:,0] = pop_list
M_STD[:,0] = pop_list

np.savetxt("M_Mean_" + MECHANISM + ".csv", M_Mean, delimiter=',', header="pop,E1,E2,logK10,logk20,a,log_diff",comments='' )      
np.savetxt("M_STD_" + MECHANISM + ".csv", M_STD, delimiter=',', header="pop,E1,E2,logK10,logk20,a,log_diff",comments='' ) 
#%%##########################################
################ FIGURE 1 ###################
#############################################
fig = plt.figure(figsize=(8,4)) 
ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.7])
ax1.scatter(all_E,all_j_exp,color = "r", s = 20)
ax1.set_xlim(1.5,1.64)
ax1.set_xticks(np.arange(1.5,1.64,0.04))
ax1.set_xlabel("$E - iR$ [V vs. RHE]")  
ax1.set_ylim(-20,80)
ax1.set_yticks(np.arange(0,80,20))
ax1.set_ylabel("$j$  [mA cm$^{-2}$]") 

ax2 = fig.add_axes([0.65, 0.2, 0.3, 0.7]) 
ax2.scatter(all_E,all_log_j_exp, color = "r", s = 20)
ax2.plot(all_E,all_log_j_exp, color = "r")
ax2.set_xlim(1.5,1.64)
ax2.set_xticks(np.arange(1.5,1.64,0.04))
ax2.set_xlabel("$E - iR$ [V vs. RHE]")  
ax2.set_ylim(-1,3)
ax2.set_yticks(np.arange(0,3,1))
ax2.set_ylabel("log $j$  [mA cm$^{-2}$]") 

axes = ax1,ax2
for idx, ax in enumerate(axes):
    ax.text(-0.35,0.95, ABC[idx], transform=ax.transAxes,size=20, weight="bold")    
plt.savefig('Fig1.png',dpi = 600)

#%%##########################################
################ FIGURE 2 ###################
#############################################
df=pd.read_csv("M_Mean_" + MECHANISM + ".csv")
par_vec_opt = df.iloc[-1,1:-1].values # the first entry is the population
log_j_theory_opt = get_log_j(par_vec_opt)
offset_log_j_theory_opt = log_j_theory_opt - get_offset(log_j_theory_opt)

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_axes([0.3, 0.2, 0.5, 0.7]) 
ax1.scatter(E,log_j_exp, color = "k")
ax1.plot(E,offset_log_j_theory_opt, color = "r", linewidth = 3)
ax1.set_xlabel("$E - iR$ [V vs. RHE]")
ax1.set_ylabel("log $j$ [mA cm$^{-2}$]")
ax1.set_xlim(1.5,1.64)
ax1.set_ylim(-1,3)
ax1.set_xticks(np.arange(1.5,1.64,0.04))
ax1.set_yticks(np.arange(0,3,1))
plt.savefig("Fig2.png", dpi = 600)

#%%##########################################
################ FIGURE 4 ###################
#############################################
ls_list = ["solid",  "dashed", "dashdot", "dotted"]
M_Mean = pd.read_csv("M_Mean_" + MECHANISM + ".csv")
M_Mean["pop"] = M_Mean["pop"].astype(int)
M_STD = pd.read_csv("M_STD_" + MECHANISM + ".csv")
E_range = np.linspace(1,2,401)
logk_range = np.linspace(-10, 10, 401)

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_axes([0.2, 0.6,0.7,0.35])
ax2 = fig.add_axes([0.2, 0.1,0.7,0.35]) 

#for idx in range(4):
for idx in range(1):
    ax1.plot(E_range, norm.pdf(E_range, loc=M_Mean.iloc[-idx-1,1], scale = M_STD.iloc[-idx-1,1]), c = "r", ls = ls_list[idx])
    ax1.plot(E_range, norm.pdf(E_range, loc=M_Mean.iloc[-idx-1,2], scale = M_STD.iloc[-idx-1,2]), c = "b", ls = ls_list[idx])
    ax2.plot(logk_range, norm.pdf(logk_range, loc=M_Mean.iloc[-idx-1,3], scale = M_STD.iloc[-idx-1,3]), c= "r", ls = ls_list[idx])
    ax2.plot(logk_range, norm.pdf(logk_range, loc=M_Mean.iloc[-idx-1,4], scale = M_STD.iloc[-idx-1,4]), c= "b", ls = ls_list[idx])
ax1.set_xlabel("$E_n$ [V vs. RHE]")
ax2.set_xlabel("log $K_1^0$, log $k_2^0$ [-]")
ax1.text(-0.2,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.2,1, "B", transform=ax2.transAxes,size=20, weight="bold")
ax1.legend(["$E_1$", "$E_2$"], loc = "upper right", frameon  = False)
ax2.legend(["$K_1^0$", "$k_2^0$"], loc = "upper left", frameon  = False)
ax1.set_ylabel("Probability Density")
ax2.set_ylabel("Probability Density")
plt.savefig("Fig4.png", dpi = 600)

#%%##########################################
################ FIGURE 3 ###################
#############################################
df=pd.read_csv("M_Mean_" + MECHANISM + ".csv")
E1,E2,logK10,logk20,a = df.iloc[-1,1:-1].values
K10 = 10**logK10
E = [num/10000 for num in range(10000,20000)]
theta0 = 1/(K10*aOH*np.exp(FRT*(E-E1))+1)
theta1 = K10*aOH*np.exp(FRT*(E-E1))/(K10*aOH*np.exp(FRT*(E-E1))+1)
dtheta0_dE = [abs(theta0[idx+1]-theta0[idx])/(E[idx+1]-E[idx]) for idx in range(len(E)-1)]
dtheta1_dE = [abs(theta1[idx+1]-theta1[idx])/(E[idx+1]-E[idx]) for idx in range(len(E)-1)]

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_axes([0.2, 0.6,0.7,0.35])
ax2 = fig.add_axes([0.2, 0.1,0.7,0.35])
ax1.text(-0.2,1, "A", transform=ax1.transAxes,size=20, weight="bold")
ax2.text(-0.2,1, "B", transform=ax2.transAxes,size=20, weight="bold")

ax1.plot(E,theta0, color = "r", linewidth = 3)
ax1.plot(E,theta1, color = "b", linewidth = 3)
ax1.set_xlabel("$E - iR$ [V vs. RHE]")
ax1.set_ylabel("$θ$ [-]")
ax1.set_xlim(1,2.1)
ax1.set_ylim(-0.1,1.1)
ax1.set_xticks(np.arange(1,2.1,0.2))
ax1.set_yticks(np.arange(0,1.1,1))
ax1.legend(["$θ_M$", "$θ_{MOH}$"], loc = "center left", frameon  = False)

ax2.plot(E[:-1], dtheta0_dE, color = "r", linewidth = 3)
ax2.plot(E[:-1], dtheta1_dE, color = "b", linewidth = 3)
ax2.set_xlabel("$E - iR$ [V vs. RHE]")
ax2.set_ylabel("$dθ/dE$ [-]")
ax2.set_xlim(1,2.1)
ax2.set_ylim(0,10.1)
ax2.set_xticks(np.arange(1,2.1,0.2))
ax2.set_yticks(np.arange(0,10.1,2))
ax2.legend(["$θ_M$", "$θ_{MOH}$"], loc = "upper left", frameon  = False)

plt.savefig("Fig3.png", dpi = 600)
