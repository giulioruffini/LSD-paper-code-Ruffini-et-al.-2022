# !pip install ipython-autotime
# !pip install numba

import matplotlib.pyplot as plt
import numpy as np
import numba
from math import exp, log, e, sqrt
from scipy.signal import savgol_filter
import multiprocessing
from operator import itemgetter


 
#######################################
#    Metropolis runner
#######################################

@numba.jit(nopython=False)
def energy(x,J,h):
    """Compute energy of lattice"""
    E = -0.5 * x@J@x - x@h
    return E


@numba.jit(nopython=False)
def update(x,J,h,kT):
    """Select random lattice point, and update with Metropolis"""
    n = J.shape[0]
    i = np.random.randint(0, n)

    dE = 2*x[i] * (np.sum(J[i,:]*x[:]) + h[i]) # 1/2 *2
    #dE = dE  
    if dE <= 0 or exp(-dE / kT) > np.random.rand(): # uniform [01)
        x[i] *= -1

@numba.jit(nopython=False)
def _random_lattice(Q):
    """ Create a random lattice of Q nodes for init"""
    x = np.zeros(Q)
    for n in range(Q): 
            x[n]=np.random.randint(0,2)*2-1
    return x

@numba.jit(nopython=False)
def run_ising(J, h, kT, Nit):
    """Run metropolis for specific kT"""
    Q =J.shape[0] # number of parcels/nodes
    x = _random_lattice(Q)
    
    M_sum = 0
    M_sq_sum = 0
    lattice_sum = x*0
    M_times_lattice_sum  = x*0
    E_sum = 0
    E_sq_sum = 0
    
    lattice_history = []
    
    for i in range(Nit):
        update(x,J,h,kT) 
        # magnetization of lattice with abs for stability:
        m = np.sum(x)
        flip = np.sign(m) # to deal with flips at low temps/hobbyhorse
        # cum_sums:
        M_sum += m * flip
        M_sq_sum += m**2 #(2 point fcn no need for flip)
        lattice_sum += x * flip #(align for ave mag > 0)
        M_times_lattice_sum  += m * x #(2 point fcn no need for flip)
        
        e = energy(x,J,h)
        E_sum += e
        E_sq_sum += e**2
        
        if i % (10*Q) == 0: # save a snapshot. chit, need parethese, % precedes *!
            lattice_history.append(x.copy()) #chit, need to copy, not ref
          
    ave_M = M_sum/Nit
    ave_lattice = lattice_sum/Nit

    global_chi = (M_sq_sum / Nit - ave_M**2)/kT
    local_chi = (M_times_lattice_sum/Nit - ave_M*ave_lattice)/kT
    
    ave_E = E_sum/Nit
    Cv = ( E_sq_sum /Nit - ave_E**2) / kT**2

    # extensive quantities are given per spin
    return ave_M/Q, global_chi/Q, ave_lattice, local_chi, ave_E/Q, Cv/Q, lattice_history



@numba.jit(nopython=False)
def run_ising2(J, h, kT, Nit):
    """Run metropolis for specific kT with link suscp"""
    Q =J.shape[0] # number of parcels/nodes
    x = _random_lattice(Q)
    
    M_sum = 0
    M_sq_sum = 0
    lattice_sum = x*0
    M_times_lattice_sum  = x*0
    E_sum = 0
    E_sq_sum = 0
    # for link suscep:
    M_times_lattice_sq_sum = np.outer(x,x)*0 
    lattice_sq_sum  = np.outer(x,x)*0
    
    lattice_history = []
    
    for i in range(Nit):
        update(x,J,h,kT) 
        # magnetization of lattice with abs for stability:
        m = np.sum(x)
        flip = np.sign(m) # to deal with flips at low temps/hobbyhorse
        # cum_sums:
        M_sum += m * flip
        M_sq_sum += m**2 #(2 point fcn no need for flip)
        lattice_sum += x * flip #(align for ave mag > 0)
        M_times_lattice_sum  += m * x #(2 point fcn no need for flip)
        # link susceptibility extension:
        outerprod = np.outer(x,x) 
        M_times_lattice_sq_sum += m*outerprod*flip # 3point
        lattice_sq_sum  += outerprod
        
        e = energy(x,J,h)
        E_sum += e
        E_sq_sum += e**2
        
        if i % (10*Q) == 0: # save a snapshot. chit, need parethese, % precedes *!
            lattice_history.append(x.copy()) #chit, need to copy, not ref
            
    ave_M = M_sum/Nit
    ave_lattice = lattice_sum/Nit

    global_chi = (M_sq_sum / Nit - ave_M**2)/kT
    local_chi = (M_times_lattice_sum/Nit - ave_M*ave_lattice)/kT
    link_chi   = ( M_times_lattice_sq_sum /Nit - ave_M* lattice_sq_sum/Nit)/kT
    np.fill_diagonal(link_chi,0)
    
    ave_E = E_sum/Nit
    Cv = ( E_sq_sum /Nit - ave_E**2) / kT**2

    # extensive quantities are given per spin
    return ave_M/Q, global_chi/Q, ave_lattice, local_chi, ave_E/Q, Cv/Q, lattice_history,link_chi 

#######################################      
def one_ising_task(task_parameters):
    """The task/function to exectute many times"""
    J, h, kT, Nit = itemgetter('J', 'h','kT','Nit')(task_parameters)

    ave_M, global_chi, ave_lattice, local_chi, ave_E, Cv, lattice_history =  run_ising(J, h, kT, Nit)
    
    # convert lattice to boolean for economy
    lattice_history = np.array((1+np.array(lattice_history))/2, dtype='bool')
    
    return ave_M, global_chi, ave_lattice, local_chi, ave_E, Cv,  lattice_history

#######################################
def run_jobs(tasks, n_processors =8):
    """"Run metropolis for list of specs/tasks"""
    J, h, kT, Nit = itemgetter('J', 'h','kT','Nit')(tasks[0])
    Ts = [t["kT"] for t in tasks]
    Q =J.shape[0]
    
    # pool the jobs
    with multiprocessing.Pool(processes=n_processors) as pool:
        out = pool.map(one_ising_task, tasks)
        pool.close()
        pool.join()
    
    #unpack data
    mags =       np.array([ out[i][0] for i in range(len(out))])
    chis =       np.array([ out[i][1] for i in range(len(out))])
    ave_lattice = np.array([ out[i][2] for i in range(len(out))]).transpose()
    local_chis = np.array([ out[i][3] for i in range(len(out))]).transpose()
    Es =         np.array([ out[i][4] for i in range(len(out))]).transpose()
    Cvs =        np.array([ out[i][5] for i in range(len(out))]).transpose()
    lattice_histories = np.array([ out[i][6] for i in range(len(out))],dtype='bool')
       
    return Ts, mags, chis, ave_lattice, local_chis, Es, Cvs,lattice_histories


#######################################    
def do_job_plots(Ts, mags, chis, ave_lattice, local_chis, Es, Cvs,Nit):
    
    Q = ave_lattice.shape[0]
    n_parcels = Q
    parcel_IDs = ["p"+str(n) for n in range(1,n_parcels+1)]
    pIDs = np.arange(Q)+1 # last minute to fix parcel ID line plots
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    im0= axs[0].imshow(ave_lattice, extent=[Ts[0],Ts[-1],1,Q+1], aspect='auto')
    axs[0].set_title(r"average spin")
    axs[0].set_xlabel(r"$T$")
    axs[0].set_ylabel(r"parcel ID")
    fig.colorbar(im0,ax=axs[0],fraction=0.045, pad=0.06,
                 label=r"$\langle \sigma_n \rangle$",location='right')

    im1= axs[1].imshow(10*np.log10(local_chis+1e-10),extent=[Ts[0],Ts[-1],1,Q+1], aspect='auto')
    axs[1].set_title(r"local susceptibility")
    axs[1].set_xlabel(r"$T$")
    axs[0].set_ylabel(r"parcel ID")
    fig.colorbar(im1,ax=axs[1],fraction=0.045, pad=0.06,
                 label=r"$\chi_{_n}$ (dB)",location='right')
    plt.show()
 
    
    # plot susceptibility per spin at kT =1
    idx = np.abs(np.array(Ts) - 1).argmin()
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    ax.plot(pIDs, local_chis[:,idx],'*')
    ax.set_title("N="+ "{:.1e}".format(Nit))
    ax.set_ylabel(r"$\chi/N$  at $T=1$")
    ax.set_xticks(range(1,n_parcels+1))
    ax.set_xticklabels(parcel_IDs,rotation=70, ha="right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    ##############
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1 = ax.plot(pIDs, np.max(local_chis,axis=1), '*',  c="k", label=r"$\chi$")
    l2 = ax2.plot(pIDs, np.array(Ts)[ np.argmax(local_chis, axis=1)], '^', c="r",  label=r"$T$")
    ax2.legend();ax.legend()
    ax2.set_xticks(range(1,n_parcels+1))
    ax2.set_xticklabels(parcel_IDs,rotation=70, ha="right")
    ax.set_xticks(range(1,n_parcels+1))
    ax.set_xticklabels(parcel_IDs,rotation=70, ha="right")
    ax.set_ylabel(r"max $\chi/N$")
    ax2.set_ylabel(r"$T$ for max")
    ax.set_title("N="+ "{:.1e}".format(Nit))
    #ax.set_xlabel("parcel ID")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    ### Create interpolated versions for plots
    magshat = savgol_filter(abs(mags), 9, 2) # window size 51, polynomial order 
    chishat = savgol_filter(chis, 9, 2) 

    ##############
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(12, 3))
    ax0b = ax0.twinx()

    l1 = ax0.scatter(Ts,mags,   c="k", s=0.4, label=r"$\langle |M| \rangle$/N")
    l4 = ax0.plot(Ts,magshat,'k')
    l2 = ax0b.scatter(Ts,chis,  c="r", s=0.4, label=r"$\chi/N$")
    l3 = ax0b.plot(Ts,chishat,'r')
    ax0b.set_yscale("log")
    ax0b.legend(handles=[l1, l2])

    ax0.set_ylabel(r"Average magnetization (per spin)    $\langle |M| \rangle$/N")
    ax0b.set_ylabel(r"Susceptibility (per spin) $\chi$/N")
    #ax.axvline(x=2 / np.log(1 + np.sqrt(2)),color='gray')
    #ax.set_title("Q ="+str(Q)+ ", N="+ "{:.1e}".format(Nit))
    ax0.set_xlabel("T")
    ax0.grid(True)

    ### Create interpolated versions for plots
    Eshat = savgol_filter(Es, 9, 2) # window size 51, polynomial order 
    Cvshat = savgol_filter(Cvs, 9, 2)

    ax1b = ax1.twinx()

    l1 = ax1.scatter(Ts,Es,   c="k", s=0.4, label=r"$\langle H \rangle$/N")
    l4 = ax1.plot(Ts,Eshat,'k')
    l2 = ax1b.scatter(Ts,Cvs,  c="r", s=0.4, label=r"$C_v/N$")
    l3 = ax1b.plot(Ts,Cvshat,'r')
    ax1b.set_yscale("log")
    ax1b.legend(handles=[l1, l2])

    ax1.set_ylabel(r"Average energy (per spin) $\langle H \rangle/N$")
    ax1b.set_ylabel(r"Specific heat (per spin) $C_v/N$")
    ax1.set_xlabel("T")
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
    
###############################################################    
#######                     Complexity analysis
###############################################################
from scipy.stats import entropy as scipyentropy

#######################################    
def entropy(the_lattice):
    """entropy of lattice numpy array in base 2. """
    value, counts = np.unique(the_lattice.flatten(), return_counts=True)
    return scipyentropy(counts, base=2)

def rho0(the_lattice):
    the_string = "".join([str(int(x)) for x in the_lattice])
    """Computes compression ratio relative to entropy l_{LZW} as described in TN000344,
    https://arxiv.org/pdf/1707.09848.pdf
    The lattice  is binary numpy array here (we convert to string)"""

    compressedstring, len_dict = compress(the_string, mode='binary', verbose=False)  # returns

    # you need n bits log(max(comp))) to describe this sequence. And there are this many: len(a)...
    # DL: "you need n bits. There are these (m) of them".

    DL = np.log2(np.log2(max(compressedstring))) + np.log2(max(compressedstring)) * len(compressedstring)

    return DL / len(the_string)

#######################################    
def rho1(the_lattice):
    lattice_string = "".join([str(int(x)) for x in the_lattice])
    
    return rho0(lattice_string) / entropy(the_lattice)
    

#######################################    
def compress(the_string, mode='binary', verbose=True):
    """Compress a *string* to a list of output symbols. Starts from two sympols, 0 and 1.
       Returns the compressed string and the length of the dictionary
       If you need to, convert first arrays to a string ,
       e.g., entry="".join([np.str(np.int(x)) for x in theArray]) """

    if mode == 'binary':
        dict_size = 2
        dictionary = {'0': 0, '1': 1}
    elif mode == 'ascii':
        # Build the dictionary for generic ascii.
        dict_size = 256
        dictionary = dict((chr(i), i) for i in range(dict_size))
    else:
        print("unrecognized mode, please use binary or ascii")
    w = ""
    result = []
    for c in the_string:
        wc = w + c

        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])
    if verbose:
        print("length of input string:", len(the_string))
        print("length of dictionary:", len(dictionary))
        print("length of result:", len(result))
    return result, len(dictionary)

#######################################    
def compute_complexity(lattice_histories):

    entropy_for_temp = []
    std_entropy_for_temp = []
    
    rho0_for_temp = []
    std_rho0_for_temp = []
    
    S = lattice_histories.shape[1]
 

    for t in range(lattice_histories.shape[0]):
        e_sum = 0
        e_sq_sum =0
        rho0_sum = 0
        rho0_sq_sum = 0
        
        for s in range(lattice_histories.shape[1]):
            lat = lattice_histories[t,s,:]
            e =  entropy(lat)
            r0= rho0(lat)
            e_sum += e 
            e_sq_sum += e**2
            rho0_sum += r0
            rho0_sq_sum += r0**2
            
            
        entropy_for_temp.append( e_sum / S)
        std_entropy_for_temp.append(e_sq_sum/S - (e_sum/S)**2)
        
        rho0_for_temp.append( rho0_sum / S)
        std_rho0_for_temp.append(rho0_sq_sum / S - (rho0_sum/S)**2)
        
    return (np.array(entropy_for_temp), np.array(std_entropy_for_temp), 
            np.array(rho0_for_temp), np.array(std_rho0_for_temp) )

#######################################    
def plot_complexities(Ts, x, y):    

    ### Create interpolated versions for plots
    xhat = savgol_filter(x, 9, 2) # window size 51, polynomial order 
    yhat = savgol_filter(y, 9, 2) 

    ##############
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1 = ax.scatter(Ts,x,   c="k", s=0.4, label=r"$\rho_0$")
    l2 = ax.plot(Ts,xhat,'k')
    
    l3 = ax2.scatter(Ts,y,  c="r", s=0.4, label=r"$\sigma_{\rho_0}$")
    l4 = ax2.plot(Ts,yhat,'r')
    

    ax.legend(handles=[l1, l3])

    ax.set_ylabel(r"$\rho_0$")
    ax.set_xlabel(r"$T$")
    ax2.set_ylabel(r"$\sigma_{\rho_0}^2$")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

#######################################    
def plot_entropies(Ts, x, y):    

    ### Create interpolated versions for plots
    xhat = savgol_filter(x, 9, 2) # window size 51, polynomial order 
    yhat = savgol_filter(y, 9, 2) 

    ##############
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1 = ax.scatter(Ts,x,   c="k", s=0.4, label=r"$h_0$")
    l2 = ax.plot(Ts,xhat,'k')
    
    l3 = ax2.scatter(Ts,y,  c="r", s=0.4, label=r"$\sigma_{h_0}$")
    l4 = ax2.plot(Ts,yhat,'r')
    

    ax.legend(handles=[l1, l3])

    ax.set_ylabel(r"$h_0$")
    ax.set_xlabel(r"$T$")
    ax2.set_ylabel(r"$\sigma_{h_0}^2$")
    ax.grid(True)
    plt.tight_layout()
    plt.show()



