import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from m_optim import eff
from m_optim7 import eff1, producer1
from m_class import Env1, Optclasspl_dis1
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
import string

np.random.seed(seed=1)
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
new_disfnames = ['results/disordered/12_01_10_10_disorderedpl07-04-2026_11:05.csv', 'results/disordered/12_01_30_10_disorderedpl07-04-2026_15:29.csv', 'results/disordered/12_01_50_10_disorderedpl_seed_108-04-2026_06:58.csv']
old_disfnames = ['results/disordered/12_01_10_10_disorderedpl16-03-2026_14:55.csv', 'results/disordered/12_01_30_10_disorderedpl16-03-2026_15:28.csv', 'results/disordered/12_01_50_10_disorderedpl19-03-2026_12:40.csv']
udisfnames = ['results/disordered/usethese/12_01_10_10_disorderedpl_singleG_18-03-2026_09:48.csv', 'results/disordered/usethese/12_01_50_10_disorderedpl_singleG_18-03-2026_09:50.csv', 'results/disordered/usethese/12_01_30_10_disorderedpl_singleG_18-03-2026_09:49.csv']
smart_fnames = ['results/disordered/12_01_10_10_disorderedpl_seed_1_smart_09-04-2026_13:34.csv', 'results/disordered/12_01_30_10_disorderedpl_seed_1_smart_09-04-2026_17:16.csv', 'results/disordered/12_01_50_10_disorderedpl_seed_1_smart_09-04-2026_17:30.csv']
smart_fnames_long = ['results/disordered/12_01_10_10_disorderedpl_seed_1_smart_10-04-2026_17:27.csv', 'results/disordered/12_01_30_10_disorderedpl_seed_1_smart_10-04-2026_07:54.csv', 'results/disordered/12_01_50_10_disorderedpl_seed_1_smart_10-04-2026_07:28.csv']
ramp_fnames = ["results/ramp/usethese/12_01_10_neg10_ramppl22-01-2026_13:40.csv", "results/ramp/usethese/12_01_30_neg10_ramppl22-01-2026_13:33.csv", "results/ramp/usethese/12_01_50_neg10_ramppl22-01-2026_13:30.csv"]
uramp_fnames = ["results/ramp/usethese/12_01_10_neg10_ramppl_singleG_22-01-2026_13:46.csv", "results/ramp/usethese/12_01_30_neg10_ramppl_singleG_22-01-2026_13:49.csv", "results/ramp/usethese/12_01_50_neg10_ramppl_singleG_22-01-2026_13:50.csv"]
ramp_fnames_05 = ['results/ramp/12_01_10_neg05_ramppl10-04-2026_14:34.csv', 'results/ramp/12_01_30_neg05_ramppl10-04-2026_14:55.csv', 'results/ramp/12_01_50_neg05_ramppl10-04-2026_15:20.csv']
uramp_fnames_05 = ['results/ramp/12_01_10_neg05_ramppl_singleG_13-04-2026_12:07.csv', 'results/ramp/12_01_30_neg05_ramppl_singleG_13-04-2026_12:08.csv', 'results/ramp/12_01_50_neg05_ramppl_singleG_13-04-2026_12:09.csv']
ramp_fnames_8 = ["results/ramp/8_01_10_neg06666666666666666_ramppl16-04-2026_14:43.csv", "results/ramp/8_01_30_neg06666666666666666_ramppl16-04-2026_14:45.csv", "results/ramp/8_01_50_neg06666666666666666_ramppl16-04-2026_14:47.csv"]
uramp_fnames_8 = ["results/ramp/8_01_10_neg06666666666666666_ramppl_singleG_16-04-2026_13:11.csv", "results/ramp/8_01_30_neg06666666666666666_ramppl_singleG_16-04-2026_13:11.csv", "results/ramp/8_01_50_neg06666666666666666_ramppl_singleG_16-04-2026_13:12.csv"]
ramp_fnames_10 = ["results/ramp/10_01_10_neg08333333333333334_ramppl16-04-2026_14:53.csv", "results/ramp/10_01_30_neg08333333333333334_ramppl16-04-2026_14:58.csv", "results/ramp/10_01_50_neg08333333333333334_ramppl16-04-2026_15:07.csv"]
uramp_fnames_10 = ["results/ramp/10_01_10_neg08333333333333334_ramppl_singleG_16-04-2026_13:12.csv", "results/ramp/10_01_30_neg08333333333333334_ramppl_singleG_16-04-2026_13:13.csv", "results/ramp/10_01_50_neg08333333333333334_ramppl_singleG_16-04-2026_13:14.csv"]
ramp_fnames_14 = ["results/ramp/14_01_10_neg11666666666666667_ramppl16-04-2026_15:47.csv", "results/ramp/14_01_30_neg11666666666666667_ramppl16-04-2026_16:22.csv", "results/ramp/14_01_50_neg11666666666666667_ramppl16-04-2026_17:15.csv"]
uramp_fnames_14 = ["results/ramp/14_01_10_neg11666666666666667_ramppl_singleG_16-04-2026_13:16.csv", "results/ramp/14_01_30_neg11666666666666667_ramppl_singleG_16-04-2026_13:18.csv", "results/ramp/14_01_50_neg11666666666666667_ramppl_singleG_16-04-2026_13:21.csv"]
dis_fnames_8 = ["results/disordered/8_01_10_10_disorderedpl_seed_1_smart_16-04-2026_11:37.csv", "results/disordered/8_01_30_10_disorderedpl_seed_1_smart_16-04-2026_14:16.csv", "results/disordered/8_01_50_10_disorderedpl_seed_1_smart_16-04-2026_17:16.csv"]
dis_fnames_10 = ["results/disordered/10_01_10_10_disorderedpl_seed_1_smart_15-04-2026_16:37.csv", "results/disordered/10_01_30_10_disorderedpl_seed_1_smart_15-04-2026_22:31.csv", "results/disordered/10_01_50_10_disorderedpl_seed_1_smart_16-04-2026_02:31.csv"]
dis_fnames_14 = ["results/disordered/14_01_10_10_disorderedpl_seed_1_smart_17-04-2026_09:38.csv", "results/disordered/14_01_30_10_disorderedpl_seed_1_smart_18-04-2026_06:35.csv", "results/disordered/14_01_50_10_disorderedpl_seed_1_smart_18-04-2026_22:56.csv"]

     
#Keep
def scidisplay(x, precision):
    """
    Convert a number to LaTeX-style scientific notation string: y × 10^{n}.

    Parameters:
        x : float or np.float
            The number to convert.
        precision : int
            Number of decimal places for the coefficient y.

    Returns:
        str: LaTeX string like '1.23 \\times 10^{-5}' ready for matplotlib.
    """
    if x == 0:
        return f"$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exponent)

    return f"${coeff:.{precision}f} \\times 10^{{{exponent}}}$"

#Keep
def generate_energy_level_diagram(eps, ax=None, **hlines_kwargs):

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8), dpi=300)
        axarg = False
    else:
        axarg = True

    N = len(eps)

    xmin = np.arange(1,N+1) - 0.45
    xmax = np.arange(1,N+1) + 0.45
    # Draw the energy levels with additional hlines parameters
    ax.hlines(y=eps, xmin=xmin, xmax=xmax, **hlines_kwargs)

    # Set the y-axis label
    ax.set_ylabel('Energy', fontsize='xx-large')

    # Set the x-axis limits and remove the x-axis ticks
    ax.set_xticks(np.arange(1, N+1), np.arange(1, N+1))
    ax.set_xlabel('Site', fontsize='xx-large')
    ax.tick_params(axis='both', labelsize='x-large')

    # Show the plot if a new figure was created
    if not axarg:
        plt.show()
        return fig, ax

#Keep
def energy_noise(fname, color1='k', color2='r', ax=None, logscale=False, ind=0):
    df = pd.read_csv(fname)
    if len(df) == 0:
        df = pd.read_csv(fname)
    if type(ax) == type(None):
        fig, ax = plt.subplots()
        axarg = False
    else:
        fig = ax.get_figure()
        axarg = True
    
    secax = ax.twinx()
    index = np.argwhere(df["Unnamed: 0"].values == ind).item()
    N = df["N"].iloc[index]
    eps = df[["e" + str(k+1) for k in range(N)]].iloc[index].values
    Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[index].values

    

    #ax.plot(nums, eps, linewidth=0, marker='_', color=color1)
    generate_energy_level_diagram(eps, ax=ax, color=color1, linewidth=4)
    if logscale:
        secax.semilogy(range(1,N+1), Gammas, linewidth=0, marker='*', color=color2, markersize=10)
    else:
        secax.plot(range(1,N+1), Gammas, linewidth=0, marker='*', color=color2, markersize=10)

    ax.set_xlabel("Site", fontsize='xx-large')
    ax.set_ylabel("Site Energy", fontsize='xx-large', color=color1)
    #secax.set_ylabel("$\\Gamma$", fontsize='xx-large', color=color2, rotation='horizontal')

    ax.tick_params(axis='both', labelsize='x-large')
    secax.tick_params(axis='y', labelsize='x-large', labelcolor=color2)
    return fig, ax, secax
    """ if not axarg:
        return fig, ax """
    
#Keep
def rhoss(Gammas, ex, lk, gleak, eps, Jmax, alpha):
    N = len(eps)
    H0 = jnp.diagflat(eps, 0)
    X, Y = jnp.meshgrid(jnp.arange(N, dtype=float), jnp.arange(N, dtype=float))
    matrix = jnp.abs(Y - X)
    #Powers for coupling strength -- make sure diagonal elements of matrix (0s) remain 0
    alphas = jnp.fill_diagonal(-alpha * jnp.ones((N, N), dtype=float), 0.0, inplace=False)
    H1 = jnp.fill_diagonal(Jmax * jnp.power(matrix, alphas), 0.0, inplace=False)
    H = H0 + H1

    Hmat1 = jnp.kron(H, jnp.eye(N))
    Hmat2 = jnp.kron(jnp.eye(N), H.T)
    mat = -1j*(Hmat1 - Hmat2)

    klambda = lambda k: Gammas[k] * kloop(k, N)
    kvmap = jax.vmap(klambda, in_axes=0)
    mat += jnp.sum(kvmap(jnp.arange(N)), axis=0)

    #Leak term
    ketN = jnp.zeros(N).at[lk].set(1.)
    ket1 = jnp.zeros(N).at[ex].set(1.)
    L = jnp.outer(ket1, ketN)
    LL = L.T @ L
    LrhoL1 = jnp.kron(L, jnp.ones((N,N)))
    LrhoL2 = jnp.kron(jnp.ones((N,N)), L.conj())
    LrhoL = LrhoL1 * LrhoL2
    LLrho = jnp.kron(LL, jnp.eye(N))
    rhoLL = jnp.kron(jnp.eye(N), LL.T)
    mat += gleak * (LrhoL - 0.5 * (LLrho + rhoLL))

    popcon = jnp.reshape(jnp.eye(N, dtype=complex), N**2)
    mat = mat.at[ex*(N+1),:].set(popcon)
    lhs = jnp.zeros((N**2,1), dtype=complex).at[ex*(N+1)].set(1.)

    rho = jsp.linalg.solve(mat, lhs).reshape((N,N))

    return rho
#Keep
def kloop(k, N):
    ket = jnp.zeros(N).at[k].set(1.)
    L = jnp.outer(ket, ket)
    LL = L.T @ L
    LrhoL1 = jnp.kron(L, jnp.ones((N,N)))
    LrhoL2 = jnp.kron(jnp.ones((N,N)), L.conj())
    LrhoL = LrhoL1 * LrhoL2
    LLrho = jnp.kron(LL, jnp.eye(N))
    rhoLL = jnp.kron(jnp.eye(N), LL.T)
    return LrhoL - 0.5*(LLrho + rhoLL)

#Keep
def singleG_plot(fname, color=None, ax=None, minP=-7, maxP=1., n=50, index=0, **kw_args):
    df = pd.read_csv(fname)
    N = df["N"].iloc[index]
    alpha = df["alpha"].iloc[index]
    Jmax = df["Jmax"].iloc[index]
    ex = df["ex"].iloc[index]
    lk = df["lk"].iloc[index]
    eps = df[["e" + str(k+1) for k in range(N)]].iloc[index].values
    gleak = df["gleak"].iloc[index]
    Gamma_opt = df["Gamma"].iloc[index]
    
    if type(color) == type(None):
        if alpha <= 1.0:
            color = 'tab:purple'
        elif alpha <= 3.0:
            color = 'tab:orange'
        else:
            color = 'tab:blue'
    
    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=(8,8))

    


    Gamma = np.sort(np.append(10.0**np.linspace(minP, maxP, n), Gamma_opt))
    etas = np.zeros(n+1)
    for k in range(n+1):
        etas[k] = eff(Gamma[k]*jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
    
    ax.semilogx(Gamma, etas, linewidth=6, color=color, label=f"$\\alpha = {int(alpha)}$", **kw_args)
    ylims = ax.get_ylim()

    ax.axvline(x=Gamma_opt, ymin=0, ymax=1, color=color, linestyle='--', linewidth=3)

    ax.set_ylabel("$\\eta$", fontsize='xx-large')
    ax.set_xlabel("$\\Gamma$", fontsize='xx-large')
    ax.tick_params(axis='both', labelsize='x-large')
    ax.legend(fontsize='x-large')
    if type(ax) == type(None):
        return fig, ax
    
#Keep: figs 3, 7
def singleG_allplots_only(fnames, ind, **kw_args):
    match_indices = np.zeros(3)
    #figsize=(5,5) for best fontsize
    fig, ax = plt.subplots(figsize=(5,5), layout='constrained', dpi=300)
    for k in range(3):
        fname = fnames[k]
        df = pd.read_csv(fname)
        index = np.argwhere(df["Unnamed: 0"].values == ind).item()
        singleG_plot(fnames[k], ax=ax, index=index, minP=-5., maxP=1., n=75, **kw_args)
        match_indices[k] = index
    ax.tick_params(axis='both', labelsize='x-large')
    ax.legend(fontsize='xx-large')
    ax.set_xlabel("$\\Gamma$", fontsize='xx-large')
    ax.set_ylabel("$\\eta$", fontsize='xx-large')
    line = ax.get_children()[5] #The last dashed line in axenaqt
    plt.setp(line, linestyle=(1.6, (3.7, 1.6)))
    return fig, ax

#Keep: figs 2, 6
def localizationplots(fnames = ["results/ramp/usethese/12_01_10_neg10_ramppl_singleG_22-01-2026_13:46.csv", "results/ramp/usethese/12_01_30_neg10_ramppl_singleG_22-01-2026_13:49.csv", "results/ramp/usethese/12_01_50_neg10_ramppl_singleG_22-01-2026_13:50.csv"], ind=0):
    
    fig, axs = plt.subplot_mosaic([["axe", "axe", "main"], ["rho1", "rho2", "rho3"]], figsize=(16,8), layout='tight')
    markers = ['s', 'o', '^']
    rhoaxes = [axs["rho1"], axs["rho2"], axs["rho3"]]
    vmins = jnp.zeros(3)
    vmaxs = jnp.zeros(3)
    rhos = []
    mycolors = []
    axe = axs["axe"]
    alphabet = string.ascii_lowercase
    ax = axs["main"]
    for k in range(len(fnames)):
        fname = fnames[k]
        df = pd.read_csv(fname) 
        index = np.argwhere(df["Unnamed: 0"].values == ind).item()  
        N = df["N"].iloc[index]
        eps = df[["e" + str(j+1) for j in range(N)]].iloc[index].values
        ex = df["ex"].iloc[index]
        lk = df["lk"].iloc[index]
        gleak = df["gleak"].iloc[index]
        Jmax = df["Jmax"].iloc[index]
        alpha = df["alpha"].iloc[index]
        generate_energy_level_diagram(eps, axe, linewidth=4, color='k')
        if alpha <= 1.0:
            color = 'tab:purple'
            cmap = 'Purples'
        elif alpha <= 3.0:
            color = 'tab:orange'
            cmap = 'Oranges'
        else:
            color = 'tab:blue'
            cmap = 'Blues'
        mycolors.append(cmap)
        rho = rhoss(jnp.zeros(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        pops = jnp.abs(jnp.diag(rho))
        vmaxs = vmaxs.at[k].set(jnp.max(jnp.abs(jnp.fill_diagonal(rho, 0.0, inplace=False))))
        vmins = vmins.at[k].set(jnp.min(jnp.abs(jnp.fill_diagonal(rho, 1.0, inplace=False))))
        rhos.append(rho)
        ax.semilogy(range(1, N+1), pops, linewidth=0, color=color, marker=markers[k], markersize=10, label=f"$\\alpha = {int(alpha)}$")
    
    
    ax.set_ylabel("Population", fontsize='xx-large')
    ax.tick_params(axis='y', labelsize='x-large')

    

    ax.tick_params(axis='x', labelsize='x-large')
    ax.set_xticks(np.arange(2, N+1, 2))

        
    vmin = vmins.min()
    vmax = vmaxs.max()
    for k in range(3):
        cmap = mycolors[k]
        rho = rhos[k]
        axrho = rhoaxes[k]
        #cmap = colors.LinearSegmentedColormap.from_list(name='my_cmap', colors=['w', color])
        im = axrho.imshow(jnp.fill_diagonal(jnp.abs(rho), 0.0, inplace=False), cmap=cmap, vmin=vmin, vmax=vmax, norm='linear')
        axrho.set_xticks(ticks=range(1,N+1,2), labels=range(2,N+2,2))
        
        cbar = fig.colorbar(im, ax=axrho)
        cbar.ax.set_ylabel("$|\\rho|$", fontsize='xx-large')
        cbar.ax.tick_params(axis='y', labelsize='x-large')
        axrho.set_xlabel("Site", fontsize='xx-large')
        axrho.set_yticks(ticks=range(1,N+1,2), labels=range(2,N+2,2))
        if k == 0:
            axrho.set_ylabel("Site", fontsize='xx-large')
            axrho.tick_params(axis='both', labelsize='x-large')
        else:
            axrho.tick_params(axis='y', labelsize=0)
            axrho.tick_params(axis='x', labelsize='x-large')
        axrho.set_title(alphabet[k+2], fontsize='xx-large', fontweight='bold', loc='left')

    
    axe.set_xlabel("")
    axe.set_xticks(range(2, N+2, 2), range(2, N+2, 2))
    axe.set_ylabel("Energy", fontsize='xx-large')
    axe.tick_params(axis='both', labelsize='x-large')
    axe.set_title("a", fontsize='xx-large', fontweight='bold', loc='left')
    #axs["main"].set_xticks(np.arange(2, N+2, 2), np.arange(2, N+2, 2))
    axs["main"].set_title("b", fontsize='xx-large', fontweight='bold', loc='left')
    axs["main"].legend(fontsize='x-large', loc='lower left')

    
    return fig, axs

#Keep
def threesite_contour_panel_traj(fnames=["results/trajectories/3_01_10_ramppl_exfix_traj_25-02-2026_11:27.csv", "results/trajectories/3_01_30_ramppl_exfix_traj_25-02-2026_11:27.csv", "results/trajectories/3_01_50_ramppl_exfix_traj_25-02-2026_11:27.csv"], minP=-7., maxP=1., n=25, figsize=(16,4), cmap='cool'):
    nf = len(fnames)
    fig, axs = plt.subplots(nrows=1, ncols=nf+1, figsize=figsize, layout='constrained')
    powers = [jnp.array([a, b]) for a in jnp.linspace(minP, maxP, n) for b in jnp.linspace(minP, maxP, n)]
    alphabet = string.ascii_lowercase
    eta_arrays = []
    alphas = []
    dfs = []
    tcolors = list(colors.TABLEAU_COLORS)
    for fname in fnames:
        df = pd.read_csv(fname)
        dfs.append(df)
        ex = df["ex"].iloc[0]
        lk = df["lk"].iloc[0]
        gleak = df["gleak"].iloc[0]
        N = df["N"].iloc[0]
        eps = df[["e" + str(k+1) for k in range(N)]].iloc[0].values
        Jmax = df["Jmax"].iloc[0]
        alpha = df["alpha"].iloc[0]
        alphas.append(alpha)
        f = lambda G: eff(jnp.concatenate((jnp.array([0.0]), 10.0**G)), ex, lk, gleak, eps, Jmax, alpha)
        
        
        fvec = jax.vmap(f)
        effs = fvec(jnp.array(powers))
        eta_arrays.append(effs)
    etaarray = jnp.array(eta_arrays)
    vmax = jnp.max(etaarray)
    vmin = jnp.min(etaarray)
    """     maxeta = jnp.unravel_index(jnp.argmax(etaarray, keepdims=True), etaarray.shape)
    mineta = jnp.unravel_index(jnp.argmin(etaarray, keepdims=True), etaarray.shape)
    maxax_ind = maxeta[0].item()
    mineta_ind = mineta[0].item() """
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ims = []
    for k in range(nf):
        ax = axs[k+1]
        if k > 0:
            ax.sharey(axs[1])
        df = dfs[k]
        effs = eta_arrays[k]
        im = ax.contourf(jnp.linspace(minP, maxP, n), jnp.linspace(minP, maxP, n), effs.reshape((n,n)).T, cmap=cmap, norm=norm, alpha=0.5)
        ims.append(im)
        ax.set_xticks(ticks=ax.get_xticks(), labels=[f"$10^{{{round(p)}}}$" for p in ax.get_xticks()], fontsize='x-large')
        ax.set_yticks(ticks=ax.get_yticks(), labels=[f"$10^{{{round(p)}}}$" for p in ax.get_yticks()], fontsize='x-large')
        if k > 0:
            ax.tick_params(axis='y', labelsize=0)
        
        for j in range(1,10):
            colour = tcolors[j-1]
            count = df["count" + str(j)].iloc[0]
            xs = np.log10(df["Gamma2_" + str(j)].values[:count])
            ys = np.log10(df["Gamma3_" + str(j)].values[:count])
            ax.plot(xs, ys, linewidth=2, color=colour, linestyle='--')
        for j in range(1,10):
            colour = tcolors[j-1]
            count = df["count" + str(j)].iloc[0]
            xs = np.log10(df["Gamma2_" + str(j)].values[:count])
            ys = np.log10(df["Gamma3_" + str(j)].values[:count])
            ax.scatter(xs[0], ys[0], marker='s', s=60, c=colour, edgecolor='k')
            ax.scatter(xs[-1], ys[-1], marker='o', s=60, c=colour, edgecolor='k')

        alpha = alphas[k]
        panel_label = alphabet[k+1]
        ax.set_title(panel_label, loc='left', fontsize='xx-large', fontweight='bold')
        ax.set_title(f"$\\alpha = {int(alpha)}$", fontsize='xx-large')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[1:], alpha=0.5)
    cbar.ax.set_ylabel("$\\eta$", fontsize='xx-large')
    cbar.ax.tick_params(axis='y', labelsize='x-large')

    axs[1].set_ylabel("$\\Gamma_3$", fontsize='xx-large')

    ind = round(np.median(range(nf)))
    axs[ind+1].set_xlabel("$\\Gamma_2$", fontsize='xx-large')

    generate_energy_level_diagram(df[["e1", "e2", "e3"]].iloc[0].values, axs[0], linewidth=4, color='k')
    axs[0].set_xlabel("Site", fontsize='xx-large')
    axs[0].set_ylabel("Energy", fontsize='xx-large')
    axs[0].tick_params(axis='both', labelsize='x-large')
    axs[0].set_title('a', fontsize='xx-large', fontweight='bold', loc='left')
    

    return fig, axs

#Keep
def disordered_histogram_good(fnames = None, all=True):
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    if fnames is None:
        fnames = ['results/disordered/12_01_10_10_disorderedpl16-03-2026_14:55.csv', 'results/disordered/12_01_30_10_disorderedpl16-03-2026_15:28.csv', 'results/disordered/12_01_50_10_disorderedpl19-03-2026_12:40.csv']
    
    dfs = [pd.read_csv(fname) for fname in fnames]
    alphabet = string.ascii_lowercase
   
    N = 12
    seg = len(dfs[0])*N
    Gamma_array = np.zeros((3*seg,2))
    for k in range(3):
        df = pd.read_csv(fnames[k])
        if not all:
            df = df.iloc[np.argwhere(df["success"].values == 1).squeeze()]

        """ for j in range(len(real_inds)):
            
            real_ind = np.argwhere(df["Unnamed: 0"].values == ind[j]).item()
            real_inds[j] = real_ind"""

        N = df["N"].iloc[0]
        alpha = df["alpha"].iloc[0]
        if alpha <= 1.:
            color = 'tab:purple'
        elif alpha <= 3.:
            color = 'tab:orange'
        else:
            color = 'tab:blue'
        Gammas = df[["Gamma" + str(k+1) for k in range(N)]].values.reshape(-1,)
        Gamma_array[k*seg:(k+1)*seg,0] = Gammas
        Gamma_array[k*seg:(k+1)*seg,1] = alpha * np.ones(seg)
        """ vals1, bins1, patches1 = ax.hist(np.log10(Gammas), bins=20, color=color, density=True)
        vals.append(vals1)
        bins.append(bins1)
        patches.append(patches1)
        ax.set_ylabel("Density", fontsize='xx-large')
        ax.tick_params(axis='y', labelsize='x-large')
        ax.set_title(alphabet[k], loc='left', fontsize='xx-large', fontweight='bold') """
    df1 = pd.DataFrame(data=Gamma_array, columns=['Gammas', 'alpha'])
    sns.histplot(data=df1, x='Gammas', hue='alpha', palette=['tab:purple', 'tab:orange', 'tab:blue'], element='step', stat='count', linewidth=2, log_scale=True, hue_order=[1, 3, 5], ax=ax, bins=50)
    ax.set_xlabel("$\\Gamma$", fontsize='xx-large')
    ticks = ax.get_xticks()
    sns.move_legend(ax, loc='best', title='', fontsize='x-large', labels=[f"$\\alpha = {k}$" for k in [1, 3, 5]])
    #ax.set_xticklabels(labels=[f"$10^{{{int(k)}}}$" for k in ticks])
    ax.tick_params(axis='both', labelsize='x-large')
    ax.set_ylabel("Count", fontsize='xx-large')


#Keep
def rhoplots_uniform_optimized(fnames, ufnames, ind):
    alphabet = string.ascii_lowercase
    fig = plt.figure(figsize=(16,8), layout='constrained', dpi=300)

    gs = GridSpec(ncols=9, nrows=2, width_ratios=[8,0.5,1.5,8,0.5,1.5,8,0.5,1.5], height_ratios=[1,1]) #every third column empty for space
    axorhos = [fig.add_subplot(gs[0,3*k]) for k in range(3)] #cols mod 0

    axurhos = [fig.add_subplot(gs[1,3*k]) for k in range(3)] #cols mod 0

    
    caxes = [fig.add_subplot(gs[:2, 3*k+1]) for k in range(3)] #cols mod 1, colorbars for orho, urho 

    vmins1 = jnp.zeros(3)
    vmaxs1 = jnp.zeros(3)
    vmins2 = jnp.zeros(3)
    vmaxs2 = jnp.zeros(3)
    alphas = jnp.zeros(3)
    rhos = []
    urhos = []
    for j in range(3):
        fname = fnames[j]
        ufname = ufnames[j]
        df = pd.read_csv(fname)
        udf = pd.read_csv(ufname)
        real_ind = np.argwhere(df["Unnamed: 0"].values == ind).item()
        N = df["N"].iloc[real_ind]
        
        eps = df[["e" + str(k+1) for k in range(N)]].iloc[real_ind].values
        alpha = df["alpha"].iloc[real_ind]
        Jmax = df["Jmax"].iloc[real_ind]
        ex = df["ex"].iloc[real_ind]
        lk = df["lk"].iloc[real_ind]
        gleak = df["gleak"].iloc[real_ind]
        Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[real_ind].values
        
        Gamma = udf["Gamma"].iloc[real_ind]

        alphas = alphas.at[j].set(alpha)

        if alpha <= 1.:
            color = 'tab:purple'
            cmap = 'Purples'
        elif alpha <= 3.:
            color = 'tab:orange'
            cmap = 'Oranges'
        else:
            color = 'tab:blue'
            cmap = 'Blues'

        label = f"$\\alpha = {int(alpha)}$"

        rho = rhoss(jnp.array(Gammas), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        urho = rhoss(Gamma * jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        vmin = min(jnp.min(jnp.abs(rho)), jnp.min(jnp.abs(urho)))
        vmins1 = vmins1.at[j].set(vmin)

        pops = jnp.abs(jnp.diagonal(rho))
        upops = jnp.abs(jnp.diagonal(urho))

        rhodiv = jnp.abs(rho) / jnp.abs(urho)

        rho = jnp.fill_diagonal(jnp.abs(rho), val=0.0, inplace=False)
        urho = jnp.fill_diagonal(jnp.abs(urho), val=0.0, inplace=False)
        vmax = max(jnp.max(rho), jnp.max(urho))
        vmaxs1 = vmaxs1.at[j].set(vmax)

       
        
        
        
        vmins2 = vmins2.at[j].set(min(jnp.min(rhodiv), jnp.min(1/rhodiv)))
        vmaxs2 = vmaxs2.at[j].set(max(jnp.max(rhodiv), jnp.max(1/rhodiv)))

        rhos.append(rho)
        urhos.append(urho)



    vmin1 = jnp.min(vmins1)
    vmax1 = jnp.max(vmaxs1)

    vmin2 = jnp.min(vmins2)
    vmax2 = jnp.max(vmaxs2)

    for j in range(3):
        alpha = alphas[j]
        if alpha <= 1.:
            color = 'tab:purple'
            cmap = 'Purples'
        elif alpha <= 3.:
            color = 'tab:orange'
            cmap = 'Oranges'
        else:
            color = 'tab:blue'
            cmap = 'Blues'
        ax = axorhos[j]
        uax = axurhos[j]
        cax = caxes[j]

        rho = rhos[j]
        urho = urhos[j]


        im = ax.imshow(rho, cmap=cmap, norm='log', vmin=vmin1, vmax=vmax1, aspect='auto')
        uim = uax.imshow(urho, cmap=cmap, norm='log', vmin=vmin1, vmax=vmax1, aspect='auto')

        
        
        cbar = fig.colorbar(im, cax=cax)
        #cax.set_ylabel("$|\\rho|$", fontsize='xx-large', rotation='horizontal')
        cax.tick_params(axis='both', labelsize='x-large')
    
        
        
    #Axes loop
    for k in range(3):
        ax = axorhos[k]
        ax.tick_params(axis='both', labelsize=0)
        title = ax.set_title("$|\\rho_o|$", fontsize='xx-large')
        ax.annotate(f"$\\alpha = {int(alphas[k])}$\n", fontsize=25, xycoords=title, xy=(0.5,1), horizontalalignment='center', verticalalignment='bottom')
        ax.set_title(alphabet[k] + "1", fontsize='xx-large', fontweight='bold', loc='left')
        uax = axurhos[k]
        uax.tick_params(axis='both', labelsize=0)
        uax.set_title("$|\\rho_u|$", fontsize='xx-large')
        uax.set_title(alphabet[k] + "2", fontsize='xx-large', fontweight='bold', loc='left')
        
        
       
       
        if k > 0:
            ax.sharey(axorhos[0])
            uax.sharey(axurhos[0])
        


    for a in [axorhos[0], axurhos[0]]:
        a.set_yticks(range(1, N+1, 2), range(2, N+2, 2))
        a.set_ylabel("Site", fontsize='xx-large')
        a.tick_params(axis='y', labelsize='x-large')
    
    axurhos[1].set_xlabel("Site", fontsize='xx-large')
    for a in axurhos:
        a.set_xticks(range(1, N+1, 2), range(2, N+2, 2))
        a.tick_params(axis='x', labelsize='x-large')
    return fig, axurhos, caxes

#Keep
def coherence_length(rho):
    N = rho.shape[0]
    pairs = [[m, n] for m in range(N) for n in range(N) if n != m]
    num = 0.
    denom = 0.
    for pair in pairs:
        m = pair[0]
        n = pair[1]
        num += np.abs(m-n)*np.abs(rho[m,n])
        denom += np.abs(rho[m,n])
    
    return num/denom
#Keep
def energy_mismatches(eps):
    if eps.ndim == 1:
        N = len(eps)
        eps = eps.reshape((1,N))
    else:
        N = eps.shape[1]
        
    deltas = np.zeros(eps.shape)

    for n in range(N-1):
        deltas[:,n] = np.abs(eps[:,n] - eps[:,n-1]) + np.abs(eps[:,n] - eps[:,n+1])
    
    deltas[:,-1] = np.abs(eps[:,-1]-eps[:,-2]) + np.abs(eps[:,-1] - eps[:,0])

    return deltas

#Keep
def resultspanel_smart(fnames, ind=0, cmap='seismic', lognorm=True):
    fig, all_axs = plt.subplots(nrows=3, ncols=3, figsize=(16,16), layout='constrained')
    vmins = np.zeros(3)
    vmaxs = np.zeros(3)
    diffrhos = []
    for k in range(3):
        axs = all_axs[k,:]
        fname = fnames[k]
        #df = filterdf(fname)
        df = pd.read_csv(fname)
        index = np.argwhere(df["Unnamed: 0"].values == ind).item()
        alpha = df["alpha"].iloc[index]
        if alpha <= 1:
            color = 'tab:purple'
        elif alpha <= 3:
            color = 'tab:orange'
        else:
            color = 'tab:blue'
        _, _, secax = energy_noise(fname, color2 = color, ax=axs[0], ind=ind)
        label = axs[0].set_ylabel(f"$\\alpha = {int(alpha)}$ \n Energy", fontsize='xx-large')
        label = secax.set_ylabel("$\\Gamma$", color=color, fontsize='xx-large', rotation='vertical', horizontalalignment='right')
        N = df["N"].iloc[index]
        Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[index].values
        eps = df[["e" + str(k+1) for k in range(N)]].iloc[index].values
        ex = df["ex"].iloc[index]
        lk = df["lk"].iloc[index]
        gleak = df["gleak"].iloc[index]
        Jmax = df["Jmax"].iloc[index]

        rho = rhoss(jnp.array(Gammas), ex, lk, gleak, jnp.array(eps), Jmax, alpha)

        constG = df["Gamma"].iloc[np.argwhere(df["Unnamed: 0"].values == ind).item()]
        rho_u = rhoss(constG * jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        pops = jnp.abs(jnp.diagonal(rho, 0))
        pops_u = jnp.abs(jnp.diagonal(rho_u, 0))

        axs[1].plot(jnp.arange(1, N+1), pops_u, linewidth=0, marker='o', markersize=10, color='k', label=f'$\\Gamma = {round(constG, 4)}$')
        axs[1].plot(jnp.arange(1, N+1), pops, linewidth=0, marker='*', markersize=10, color=color, label='Optimized')
        axs[1].set_xlabel("Site", fontsize='xx-large')
        axs[1].set_ylabel("Population", fontsize='xx-large')
        axs[1].tick_params(axis='both', labelsize='x-large')
        axs[1].legend(fontsize='xx-large')

        axs[0].set_xticks(ticks=jnp.arange(2, N+2, 2), labels=jnp.arange(2, N+2, 2))
        axs[1].set_xticks(ticks=jnp.arange(2, N+2, 2), labels=jnp.arange(2, N+2, 2))
        
        if k < 2:
            for ax in axs:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelsize=0)

        diffrho = jnp.abs(rho) / jnp.abs(rho_u)
        diffrhos.append(diffrho)
        vmin = 10**jnp.min(-1.* jnp.abs(jnp.log10(diffrho)))
        vmax = 10**jnp.max(jnp.abs(jnp.log10(diffrho)))
        vmins[k] = vmin
        vmaxs[k] = vmax
    
    vmin = np.min(vmins)
    vmax = np.max(vmaxs)
    for k in range(3):
        axs = all_axs[k]
        diffrho = diffrhos[k]
        diffrho = jnp.fill_diagonal(diffrho, 1.0, inplace=False)
        """ if vmin > 0.1:
            vmin = 0.1
            vmax = 10. """
        if lognorm:
            im = axs[2].imshow(diffrho, cmap=cmap, norm=colors.SymLogNorm(linthresh = vmin.item()/10., linscale=1., vmin=vmin, vmax=vmax))
        else:
            im = axs[2].imshow(diffrho, cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=axs[2])
        
        axs[2].set_ylabel("Site", fontsize='xx-large')
        
            
        axs[2].set_xticks(ticks=range(0,N), labels=range(1,N+1))
        axs[2].set_yticks(ticks=range(0,N), labels=range(1,N+1))
        cbar.ax.set_ylabel("$|\\rho_o| / |\\rho_u|$", fontsize='xx-large')
        if k == 2:
            axs[2].tick_params(axis='both', labelsize='x-large')
            axs[2].set_xlabel("Site", fontsize='xx-large')
        else:
            axs[2].tick_params(axis='y', labelsize='x-large')
            axs[2].tick_params(axis='x', labelsize=0)
        cbar.ax.tick_params(axis='y', labelsize='x-large')


    alphabet = string.ascii_lowercase
    for k in range(len(all_axs.reshape(-1,))):
        ax = np.reshape(all_axs, (-1,))[k]
        ax.set_title(alphabet[k], fontsize='xx-large', fontweight='bold', loc='left')
    
    return fig, all_axs

#Keep
def delta_Gamma_bin(fnames, all=True):
    nf = len(fnames)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), layout='constrained', dpi=300)
    axs = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
    colours = [sns.color_palette()[4], sns.color_palette()[1], sns.color_palette()[0]]
    alphabet = string.ascii_lowercase
    for k in range(nf):
        ax = axs[k+1]
        fname = fnames[k]
        df = pd.read_csv(fname)
        #df = df.iloc[np.argwhere(df["success"].values == 1).squeeze()]
        N = df["N"].iloc[0]
        alpha = df["alpha"].iloc[0]
        Jmax = df["Jmax"].iloc[0]
        ex = df["ex"].iloc[0]
        lk = df["lk"].iloc[0]
        gleak = df["gleak"].iloc[0]
        
        lens = energy_mismatches(df[["e" + str(j+1) for j in range(1, 11)]].values)
        lens = lens.reshape(-1)
        Gammas = df[["Gamma" + str(j+1) for j in range(1, 11)]].values
        Gammas = Gammas.reshape(-1)
        hist, bin_edges = np.histogram(lens, bins=np.arange(0.0, 2.20, 0.2))
        bin_width = bin_edges[1] - bin_edges[0]
        nbins = len(bin_edges) - 1
        binned_inds = np.digitize(lens, bin_edges[:-1])
        #Gammas = df[["Gamma" + str(j+1) for j in range(N)]].values.reshape(-1)
        for j in range(nbins):
            jGammas = Gammas[binned_inds == j+1]
            #sns.pointplot(x=j, y=jGammas, errorbar='sd', color=colours[k], ax=axs[k])
            sns.boxplot(x=bin_edges[j]+0.5*bin_width, y=jGammas, color=colours[k], ax=axs[k+1])
        ax.set_ylabel("$\\Gamma$", fontsize='xx-large')
        ax.set_title(f"$\\alpha = {{{int(alpha)}}}$", fontsize='xx-large')
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize=0)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(ticks=range(nbins), labels=[round((bin_edges[m]+0.5*bin_width),3) for m in range(nbins)])
        ax.set_title(alphabet[k+1], fontsize='xx-large', fontweight='bold', loc='left')
        if k >= nf-2:
            ax.set_xlabel("Energy Mismatch", fontsize='xx-large')
            ax.tick_params(axis='x', labelsize='x-large')
    sns.histplot(lens.reshape(-1), color='tab:gray', linewidth=2, ax=axs[0], element='step')
    axs[0].set_ylabel(axs[0].get_ylabel(), fontsize='xx-large')
    axs[0].set_xlabel("")
    axs[0].set_xticks([round((bin_edges[m]+0.5*bin_width),3) for m in range(nbins)])
    axs[0].tick_params(axis='x', labelsize=0)
    axs[0].tick_params(axis='y', labelsize='x-large')
    axs[0].set_title(alphabet[0], fontsize='xx-large', fontweight='bold', loc='left')
    axs[0].annotate("", xy=(lens.mean(), 0), xytext=(lens.mean(),-0.1*(axs[0].get_ylim()[1] - axs[k].get_ylim()[0])), arrowprops=dict(facecolor='tab:gray', shrink=0.05))
    axs[0].set_title("$\\delta_n^{(\\mathrm{loc})}$", fontsize='xx-large')
    return fig, axs


#Keep
def coherence_length_eta_hist_smart(fnames, all=True):
    nf = len(fnames)
    alphabet = string.ascii_lowercase
    fig, all_axs = plt.subplots(ncols=2, nrows=nf, figsize=(8, 12), layout='constrained', dpi=300)
    axs = all_axs[:,0]
    for ax in axs:
        ax.sharex(axs[-1])
    eaxs = all_axs[:,1]
    """ for ax in eaxs:
        ax.sharex(eaxs[-1]) """
    for ax, eax in zip(axs, eaxs):
        eax.sharey(ax)
    colours = [sns.color_palette()[4], sns.color_palette()[1], sns.color_palette()[0]]
    ucolours = ['tab:gray', 'tab:gray', 'tab:gray']
    for k in range(nf):
        fname = fnames[k]
        df = pd.read_csv(fname)
        df.sort_values(by='Unnamed: 0', inplace=True)
        N = df["N"].iloc[0]
        ex = df["ex"].iloc[0]
        lk = df["lk"].iloc[0]
        gleak = df["gleak"].iloc[0]
        Jmax = df["Jmax"].iloc[0]
        alpha = df["alpha"].iloc[0]
        lens = np.zeros((len(df),2))
        if "cohlength" not in df.columns or "ucohlength" not in df.columns:
            for j in range(len(df)):
                eps = df[["e" + str(l+1) for l in range(N)]].iloc[j].values
                Gammas = df[["Gamma" + str(l+1) for l in range(N)]].iloc[j].values
                Gamma_opt = df["Gamma"].iloc[j]
            
                rho = rhoss(jnp.array(Gammas), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
                urho = rhoss(Gamma_opt*jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)

                lens[j,0] = coherence_length(rho)
                lens[j,1] = coherence_length(urho)


        
            df.insert(len(df.columns), "cohlength", lens[:,0], allow_duplicates=True)
            df.insert(len(df.columns), "ucohlength", lens[:,1], allow_duplicates=True)
            df = df.loc[:,~df.columns.duplicated()].copy()
            df.to_csv(fname)
        else:
            lens[:,0] = df["cohlength"].values
            lens[:,1] = df["ucohlength"].values

        if "ueta" not in df.columns:
            etas = np.array([eff1(df["Gamma"].iloc[j], ex, lk, gleak, jnp.array(df[["e" + str(l+1) for l in range(N)]].iloc[j].values), Jmax, alpha) for j in range(len(df))])
            df["ueta"] = etas
            df.to_csv(fname)
        
        if all:
            df1 = pd.DataFrame(lens, columns=["Optimized", "Uniform"])
            df2 = pd.DataFrame(np.concatenate((df["eta"].map(lambda x: np.abs(x)).values.reshape((-1,1)), df["ueta"].map(lambda x: np.abs(x)).values.reshape((-1,1))), axis=1), columns=["Optimized", "Uniform"])
        else:
            df1 = pd.DataFrame(lens[df["success"].values == 1.,:], columns=['Optimized', 'Uniform'])
            df2 = pd.DataFrame(np.concatenate((df["eta"].map(lambda x: np.abs(x)).values[np.argwhere(df["success"].values == 1)].reshape((-1,1)), df["ueta"].map(lambda x: np.abs(x)).values[np.argwhere(df["success"].values == 1)].reshape((-1,1))), axis=1), columns=["Optimized", "Uniform"])
        
        #axs[k].axvline(x=df1["Optimized"].mean(), ymin=0, ymax=1, color=colours[k], linestyle='--', linewidth=2)
        #axs[k].axvline(x=df1["Uniform"].mean(), ymin=0, ymax=1, color=ucolours[k], linestyle='--', linewidth=2)
        sns.histplot(data=df1, ax=axs[k], alpha=0.9, linewidth=2, element='step', palette=[colours[k], ucolours[k]], hue_order=["Optimized","Uniform"])
        axs[k].legend(labels=["Uniform", "Optimized"], fontsize='x-large') #Somehow the "first" handle is the most recently plotted with seaborn, so need to flip label order for legend call
        axs[k].tick_params(axis='both', labelsize='x-large')
        #axs[k].set_title(f"$\\alpha = {{{int(alpha)}}}$", fontsize='xx-large')
        axs[k].set_title(alphabet[k]+"1", fontsize='xx-large', fontweight='bold', loc='left')
        axs[k].set_ylabel(f"$\\alpha = {{{int(alpha)}}}$\n" + axs[k].get_ylabel(), fontsize='xx-large')
        axs[k].get_legend().remove()
        df2 = pd.DataFrame(np.concatenate((df["eta"].map(lambda x: np.abs(x)).values.reshape((-1,1)), df["ueta"].map(lambda x: np.abs(x)).values.reshape((-1,1))), axis=1), columns=["Optimized", "Uniform"])
        #eaxs[k].axvline(x=np.abs(df["eta"].mean()), ymin=0, ymax=1, color=colours[k], linestyle='--', linewidth=2)
        #eaxs[k].axvline(x=np.abs(df["ueta"].mean()), ymin=0, ymax=1, color=ucolours[k], linestyle='--', linewidth=2)
        sns.histplot(data=df2, ax=eaxs[k], alpha=0.9, linewidth=2, element='step', palette=[colours[k], ucolours[k]], hue_order=["Optimized","Uniform"])#, binwidth=0.0001)
        #eaxs[k].legend(labels=["Uniform", "Optimized"], fontsize='x-large') #Somehow the "first" handle is the most recently plotted with seaborn, so need to flip label order for legend call
        #eaxs[k].set_title(f"$\\alpha = {{{int(alpha)}}}$", fontsize='xx-large')
        eaxs[k].set_title(alphabet[k]+"2", fontsize='xx-large', fontweight='bold', loc='left')
        eaxs[k].tick_params(axis='y', labelsize=0)
        eaxs[k].set_ylabel(f"$\\alpha = {{{int(alpha)}}}$\n" + eaxs[k].get_ylabel(), fontsize=0)
        eaxs[k].tick_params(axis='x', labelsize='x-large')
        #eaxs[k].get_legend().remove()
        eaxs[k].annotate("", xy=(df2["Optimized"].mean(), 0), xytext=(df2["Optimized"].mean(), -0.1*(eaxs[k].get_ylim()[1] - eaxs[k].get_ylim()[0])), arrowprops=dict(facecolor=colours[k], shrink=0.05))
        eaxs[k].annotate("", xy=(df2["Uniform"].mean(), 0), xytext=(df2["Uniform"].mean(),-0.1*(eaxs[k].get_ylim()[1] - eaxs[k].get_ylim()[0])), arrowprops=dict(facecolor=ucolours[k], shrink=0.05))
        axs[k].annotate("", xy=(df1["Uniform"].mean(), 0), xytext=(df1["Uniform"].mean(),-0.1*(axs[k].get_ylim()[1] - axs[k].get_ylim()[0])), arrowprops=dict(facecolor=ucolours[k], shrink=0.05))
        axs[k].annotate("", xy=(df1["Optimized"].mean(), 0), xytext=(df1["Optimized"].mean(), -0.1*(axs[k].get_ylim()[1] - axs[k].get_ylim()[0])), arrowprops=dict(facecolor=colours[k], shrink=0.05))
        
    #axs[-1].set_xlabel("$\\ell_{\\mathrm{coh}}$", fontsize='xx-large')
    #eaxs[-1].set_xlabel("$\\eta$", fontsize='xx-large')
    axs[0].set_title("Coherence Length", fontsize='xx-large')
    eaxs[0].set_title("Population Flux", fontsize='xx-large')
    axs[-1].set_xlabel("$\\ell$", fontsize='xx-large')
    eaxs[-1].set_xlabel("$\\eta$", fontsize='xx-large')
    for ax in eaxs:
        ax.ticklabel_format(axis='x', scilimits=(-3,-3))
    return fig, axs


#Keep
def ell_eta_bin_ratio(fnames, plot_hist=False):
    nf = len(fnames)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10/3), layout='constrained', dpi=300)
    colours = [sns.color_palette()[4], sns.color_palette()[1], sns.color_palette()[0]]
    alphabet = string.ascii_lowercase
    bins = [np.arange(0.9, 1.2, 0.05), np.arange(0.9, 1.7, 0.15), np.arange(0.9, 1.7, 0.15)]
    for k in range(nf):
        ax = axs[k]
        fname = fnames[k]
        df = pd.read_csv(fname)
        #df = df.iloc[np.argwhere(df["success"].values == 1).squeeze()]
        N = df["N"].iloc[0]
        alpha = df["alpha"].iloc[0]
        lratios = df["cohlength"].values/df["ucohlength"].values
        hist, bin_edges = np.histogram(lratios, bins=bins[k])
        print(hist)
        hist = np.array(hist)
        bin_widths = np.diff(bin_edges)
        nbins = len(bin_edges) - 1
        binned_inds = np.digitize(lratios, bin_edges) #The last bin will be all the values that fall outside my desired histogram range.
        eratios = np.abs(df["eta"].values/df["ueta"].values)
        df1 = pd.DataFrame(data=np.concatenate((eratios.reshape(-1,1), binned_inds.reshape(-1,1)), axis=1), columns=["eratios", "inds"])
        to_drop = np.argwhere(np.logical_or(df1["inds"].values == 0, df1["inds"].values == np.max(binned_inds))).squeeze()
        if to_drop.size == 1:
            to_drop = to_drop.item()
        df1.drop(index=to_drop, inplace=True)
        df1["inds"] = df1["inds"].map(lambda x: bin_edges[int(x)-1]+0.5*bin_widths[int(x)-1])
        
        sns.boxplot(df1, x='inds', y='eratios', ax=ax, color=colours[k], native_scale=True)
        
        if plot_hist:
            secax = ax.twinx()
            sns.histplot(data=lratios, bins=bin_edges, ax=secax, color=colours[k], element='step', fill=False)
        ax.set_ylabel("")
        ax.set_title(f"$\\alpha = {{{int(alpha)}}}$", fontsize='xx-large')
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelrotation=45, labelsize='x-large')
        ax.set_xticks(ticks=bin_edges, labels=[round(m,2) for m in bin_edges])
        ax.set_title(alphabet[k], fontsize='xx-large', fontweight='bold', loc='left')
        ax.set_xlabel("$\\ell_\\mathrm{o}/\\ell_\\mathrm{u}$", fontsize='xx-large')
    
    axs[0].set_ylabel("$\\eta_\\mathrm{o}/\\eta_\\mathrm{u}$", fontsize='xx-large')
        
    
    return fig, axs

#Keep
def ramp_scaling(fnames1=ramp_fnames_8, fnames2=ramp_fnames_10, fnames3=ramp_fnames, fnames4=ramp_fnames_14):

    fig, all_axs = plt.subplots(nrows=2, ncols=2, dpi=300, layout='constrained', figsize=(10, 10), sharex=True)
    axs = all_axs[0,:]
    axs2 = all_axs[1,:]
    #fig2, axs2 = plt.subplots(nrows=2, ncols=1, dpi=300, layout='constrained', figsize=(5,10), sharex=True)
    l1s = np.zeros((3,4)) #coh lengths avg for all four N for alpha = 1
    l3s = np.zeros((3,4)) #alpha = 3
    l5s = np.zeros((3,4)) #alpha = 5
    e1s = np.zeros((3,4))
    e3s = np.zeros((3,4))
    e5s = np.zeros((3,4))
    colours = ['tab:purple', 'tab:orange', 'tab:blue']
    all_fnames = [fnames1, fnames2, fnames3, fnames4]
    all_lengths = [l1s, l3s, l5s]
    all_es = [e1s, e3s, e5s]
    Ns = np.zeros(4)
    for k in range(4):
        fnames = all_fnames[k]
        for (fname, i) in zip(fnames, range(3)):
            df = pd.read_csv(fname)
            alpha = df["alpha"].iloc[0]
            N = df["N"].iloc[0]
            if i == 0:
                Ns[k] = N
            if "cohlength" not in df.columns:
                
                Jmax = df["Jmax"].iloc[0]
                ex = df["ex"].iloc[0]
                lk = df["lk"].iloc[0]
                gleak = df["gleak"].iloc[0]
                lens = np.zeros(len(df))
                ulens = np.zeros(len(df))
                for j in range(len(df)):
                    eps = jnp.array(df[["e" + str(m+1) for m in range(N)]].iloc[j].values)
                    Gammas = jnp.array(df[["Gamma" + str(m+1) for m in range(N)]].iloc[j].values)
                    rho = rhoss(Gammas, ex, lk, gleak, eps, Jmax, alpha)
                    lens[j] = coherence_length(rho)
                    urho = rhoss(df["Gamma"].iloc[j]*jnp.ones(N), ex, lk, gleak, eps, Jmax, alpha)
                    ulens[j] = coherence_length(urho)
                df.insert(len(df.columns), "cohlength", lens)
                df.insert(len(df.columns), "ucohlength", ulens)
                df.to_csv(fname)
            
            df["eta"] = df["eta"].map(lambda x: np.abs(x))
            df["ueta"] = df["ueta"].map(lambda x: np.abs(x))
            index = df["eta"].argmax()
            all_lengths[i][0,k] = (df["cohlength"].iloc[index]/df["ucohlength"].iloc[index])
            all_lengths[i][1,k] = df["cohlength"].iloc[index]
            all_lengths[i][2,k] = df["ucohlength"].iloc[index]
            all_es[i][0,k] = (df["eta"].iloc[index]/df["ueta"].iloc[index])
            all_es[i][1,k] = np.abs(df["eta"].iloc[index])
            all_es[i][2,k] = np.abs(df["ueta"].iloc[index])
    
    """ for k in range(3):
        ax = axs[k]
        #ax.plot(Ns, l1s[k,:], label='alpha=1', color='tab:purple', marker='o')
        #ax.plot(Ns, l3s[k,:], label='alpha=3', color='tab:orange', marker='s')
        ax.plot(Ns, l5s[k,:], label='alpha=5', color='tab:blue', marker='^')
        ax.legend()
        ax.set_ylabel("Coherence length") """
    """ axs[0].set_title("Optimized/Uniform")
    axs[1].set_title("Optimized")
    axs[2].set_title("Uniform") """

    axs2[0].plot(Ns, l1s[0,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10)
    axs2[1].plot(Ns, e1s[0,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10)

    axs2[0].plot(Ns, l3s[0,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10)
    axs2[1].plot(Ns, e3s[0,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10)
    
    axs2[0].plot(Ns, l5s[0,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10)
    axs2[1].plot(Ns, e5s[0,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10)

    axs2[0].legend(fontsize='x-large')
    axs2[0].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[1].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[0].set_ylabel("$\\ell_\\mathrm{o}/\\ell_\\mathrm{u}$", fontsize='xx-large')
    axs2[1].set_ylabel("$\\eta_\\mathrm{o}/\\eta_\\mathrm{u}$", fontsize='xx-large')

    
    axs2[0].tick_params(axis='both', labelsize='x-large')
    axs2[1].tick_params(axis='both', labelsize='x-large')

    axs[0].plot(Ns, l1s[1,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10)
    axs[1].plot(Ns, e1s[1,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10)

    axs[0].plot(Ns, l3s[1,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10)
    axs[1].plot(Ns, e3s[1,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10)
    
    axs[0].plot(Ns, l5s[1,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10)
    axs[1].plot(Ns, e5s[1,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10)
    #axs[0].legend(fontsize='x-large')
    axs[0].plot(Ns, l1s[2,:], label='$\\alpha = 1$', color='tab:purple', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e1s[2,:], label='$\\alpha = 1$', color='tab:purple', marker='^', markersize=10, linestyle='--')

    axs[0].plot(Ns, l3s[2,:], label='$\\alpha = 3$', color='tab:orange', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e3s[2,:], label='$\\alpha = 3$', color='tab:orange', marker='^', markersize=10, linestyle='--')
    
    axs[0].plot(Ns, l5s[2,:], label='$\\alpha = 3$', color='tab:blue', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e5s[2,:], label='$\\alpha = 3$', color='tab:blue', marker='^', markersize=10, linestyle='--')

    axs2[0].set_title("c", fontsize='xx-large', fontweight='bold', loc='left')
    axs2[1].set_title("d", fontsize='xx-large', fontweight='bold', loc='left')
    axs[0].set_title("a", fontsize='xx-large', fontweight='bold', loc='left')
    axs[1].set_title("b", fontsize='xx-large', fontweight='bold', loc='left')

    for ax in axs:
        ax.tick_params(axis='both', labelsize='x-large')
        ax.set_xlabel("Number of Sites", fontsize='xx-large')
    
    custom_lines = [mpl.lines.Line2D([0], [0], color='k'), mpl.lines.Line2D([0], [0], color='k', ls='--')]
    axs[1].legend(custom_lines, ['Optimized', 'Uniform'], fontsize='x-large')

    axs[0].set_ylabel("Coherence Length", fontsize='xx-large')
    axs[1].set_ylabel("Population Flux", fontsize='xx-large')

#Keep
def dis_scaling(fnames1=dis_fnames_8, fnames2=dis_fnames_10, fnames3=smart_fnames_long, fnames4=dis_fnames_14):

    fig, all_axs = plt.subplots(nrows=2, ncols=2, dpi=300, layout='constrained', figsize=(10, 10), sharex=True)
    axs = all_axs[0,:]
    axs2 = all_axs[1,:]
    #fig2, axs2 = plt.subplots(nrows=2, ncols=1, dpi=300, layout='constrained', figsize=(5,10), sharex=True)
    l1s = np.zeros((3,4)) #coh lengths avg for all four N for alpha = 1
    l3s = np.zeros((3,4)) #alpha = 3
    l5s = np.zeros((3,4)) #alpha = 5
    e1s = np.zeros((3,4))
    e3s = np.zeros((3,4))
    e5s = np.zeros((3,4))

    l1s_err = np.zeros((3,4))
    l3s_err = np.zeros((3,4))
    l5s_err = np.zeros((3,4))

    e1s_err = np.zeros((3,4))
    e3s_err = np.zeros((3,4))
    e5s_err = np.zeros((3,4))

    colours = ['tab:purple', 'tab:orange', 'tab:blue']
    all_fnames = [fnames1, fnames2, fnames3, fnames4]
    all_lengths = [l1s, l3s, l5s]
    all_es = [e1s, e3s, e5s]
    all_lerrs = [l1s_err, l3s_err, l5s_err]
    all_eerrs = [e1s_err, e3s_err, e5s_err]
    Ns = np.zeros(4)
    for k in range(4):
        fnames = all_fnames[k]
        for (fname, i) in zip(fnames, range(3)):
            df = pd.read_csv(fname)
            alpha = df["alpha"].iloc[0]
            N = df["N"].iloc[0]
            if i == 0:
                Ns[k] = N
            if "cohlength" not in df.columns:
                
                Jmax = df["Jmax"].iloc[0]
                ex = df["ex"].iloc[0]
                lk = df["lk"].iloc[0]
                gleak = df["gleak"].iloc[0]
                lens = np.zeros(len(df))
                ulens = np.zeros(len(df))
                for j in range(len(df)):
                    eps = jnp.array(df[["e" + str(m+1) for m in range(N)]].iloc[j].values)
                    Gammas = jnp.array(df[["Gamma" + str(m+1) for m in range(N)]].iloc[j].values)
                    rho = rhoss(Gammas, ex, lk, gleak, eps, Jmax, alpha)
                    lens[j] = coherence_length(rho)
                    urho = rhoss(df["Gamma"].iloc[j]*jnp.ones(N), ex, lk, gleak, eps, Jmax, alpha)
                    ulens[j] = coherence_length(urho)
                df.insert(len(df.columns), "cohlength", lens)
                df.insert(len(df.columns), "ucohlength", ulens)
                df.to_csv(fname)
            
            df["eta"] = df["eta"].map(lambda x: np.abs(x))
            
            all_lengths[i][0,k] = (df["cohlength"]/df["ucohlength"]).mean()
            all_lengths[i][1,k] = df["cohlength"].mean()
            all_lengths[i][2,k] = df["ucohlength"].mean()
            all_es[i][0,k] = (df["eta"]/df["ueta"]).mean()
            all_es[i][1,k] = np.abs(df["eta"].mean())
            all_es[i][2,k] = np.abs(df["ueta"].mean())
    
            all_lerrs[i][0,k] = (df["cohlength"]/df["ucohlength"]).std()
            all_lerrs[i][1,k] = df["cohlength"].std()
            all_lerrs[i][2,k] = df["ucohlength"].std()

            all_eerrs[i][0,k] = (df["eta"]/df["ueta"]).std()
            all_eerrs[i][1,k] = df["eta"].std()
            all_eerrs[i][2,k] = df["ueta"].std()

    """ for k in range(3):
        ax = axs[k]
        #ax.plot(Ns, l1s[k,:], label='alpha=1', color='tab:purple', marker='o')
        #ax.plot(Ns, l3s[k,:], label='alpha=3', color='tab:orange', marker='s')
        ax.plot(Ns, l5s[k,:], label='alpha=5', color='tab:blue', marker='^')
        ax.legend()
        ax.set_ylabel("Coherence length") """
    """ axs[0].set_title("Optimized/Uniform")
    axs[1].set_title("Optimized")
    axs[2].set_title("Uniform") """

    axs2[0].errorbar(Ns, l1s[0,:], yerr = l1s_err[0,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10, capsize=5)
    axs2[1].errorbar(Ns, e1s[0,:], yerr = e1s_err[0,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10, capsize=5)

    axs2[0].errorbar(Ns, l3s[0,:], yerr = l3s_err[0,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10, capsize=5)
    axs2[1].errorbar(Ns, e3s[0,:], yerr = e3s_err[0,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10, capsize=5)
    
    axs2[0].errorbar(Ns, l5s[0,:], yerr = l5s_err[0,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10, capsize=5)
    axs2[1].errorbar(Ns, e5s[0,:], yerr = e5s_err[0,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10, capsize=5)

    axs2[0].legend(fontsize='x-large')
    axs2[0].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[1].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[0].set_ylabel("$\\langle \\ell_\\mathrm{o}/\\ell_\\mathrm{u}\\rangle$", fontsize='xx-large')
    axs2[1].set_ylabel("$\\langle\\eta_\\mathrm{o}/\\eta_\\mathrm{u}\\rangle$", fontsize='xx-large')

    
    axs2[0].tick_params(axis='both', labelsize='x-large')
    axs2[1].tick_params(axis='both', labelsize='x-large')

    axs[0].errorbar(Ns, l1s[1,:], yerr = l1s_err[1,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10, capsize=5)
    axs[1].errorbar(Ns, e1s[1,:], yerr = e1s_err[1,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10, capsize=5)

    axs[0].errorbar(Ns, l3s[1,:], yerr = l3s_err[1,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10, capsize=5)
    axs[1].errorbar(Ns, e3s[1,:], yerr = e3s_err[1,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10, capsize=5)
    
    axs[0].errorbar(Ns, l5s[1,:], yerr = l5s_err[1,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10, capsize=5)
    axs[1].errorbar(Ns, e5s[1,:], yerr = e5s_err[1,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10, capsize=5)
    #axs[0].legend(fontsize='x-large')
    axs[0].errorbar(Ns, l1s[2,:], yerr = l1s_err[2,:], label='$\\alpha = 1$', color='tab:purple', marker='s', markersize=10, linestyle='--', alpha=0.5, capsize=5)
    axs[1].errorbar(Ns, e1s[2,:], yerr = e1s_err[2,:], label='$\\alpha = 1$', color='tab:purple', marker='^', markersize=10, linestyle='--', alpha=0.5, capsize=5)

    axs[0].errorbar(Ns, l3s[2,:], yerr = l3s_err[2,:], label='$\\alpha = 3$', color='tab:orange', marker='s', markersize=10, linestyle='--', alpha=0.5, capsize=5)
    axs[1].errorbar(Ns, e3s[2,:], yerr = e3s_err[2,:], label='$\\alpha = 3$', color='tab:orange', marker='^', markersize=10, linestyle='--', alpha=0.5, capsize=5)
    
    axs[0].errorbar(Ns, l5s[2,:], yerr = l5s_err[2,:], label='$\\alpha = 3$', color='tab:blue', marker='s', markersize=10, linestyle='--', alpha=0.5, capsize=5)
    axs[1].errorbar(Ns, e5s[2,:], yerr = e5s_err[2,:], label='$\\alpha = 3$', color='tab:blue', marker='^', markersize=10, linestyle='--', alpha=0.5, capsize=5)

    axs2[0].set_title("c", fontsize='xx-large', fontweight='bold', loc='left')
    axs2[1].set_title("d", fontsize='xx-large', fontweight='bold', loc='left')
    axs[0].set_title("a", fontsize='xx-large', fontweight='bold', loc='left')
    axs[1].set_title("b", fontsize='xx-large', fontweight='bold', loc='left')


    """ axs2[0].plot(Ns, l1s[0,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10)
    axs2[1].plot(Ns, e1s[0,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10)

    axs2[0].plot(Ns, l3s[0,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10)
    axs2[1].plot(Ns, e3s[0,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10)
    
    axs2[0].plot(Ns, l5s[0,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10)
    axs2[1].plot(Ns, e5s[0,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10)

    axs2[0].legend(fontsize='x-large')
    axs2[0].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[1].set_xlabel("Number of sites", fontsize='xx-large')
    axs2[0].set_ylabel("$\\langle \\ell_\\mathrm{o}/\\ell_\\mathrm{u}\\rangle$", fontsize='xx-large')
    axs2[1].set_ylabel("$\\langle\\eta_\\mathrm{o}/\\eta_\\mathrm{u}\\rangle$", fontsize='xx-large')

    
    axs2[0].tick_params(axis='both', labelsize='x-large')
    axs2[1].tick_params(axis='both', labelsize='x-large')

    axs[0].plot(Ns, l1s[1,:], label='$\\alpha = 1$', color='tab:purple', marker='o', markersize=10)
    axs[1].plot(Ns, e1s[1,:], label='$\\alpha = 1$', color='tab:purple', marker='*', markersize=10)

    axs[0].plot(Ns, l3s[1,:], label='$\\alpha = 3$', color='tab:orange', marker='o', markersize=10)
    axs[1].plot(Ns, e3s[1,:], label='$\\alpha = 3$', color='tab:orange', marker='*', markersize=10)
    
    axs[0].plot(Ns, l5s[1,:], label='$\\alpha = 5$', color='tab:blue', marker='o', markersize=10)
    axs[1].plot(Ns, e5s[1,:], label='$\\alpha = 5$', color='tab:blue', marker='*', markersize=10)
    #axs[0].legend(fontsize='x-large')
    axs[0].plot(Ns, l1s[2,:], label='$\\alpha = 1$', color='tab:purple', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e1s[2,:], label='$\\alpha = 1$', color='tab:purple', marker='^', markersize=10, linestyle='--')

    axs[0].plot(Ns, l3s[2,:], label='$\\alpha = 3$', color='tab:orange', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e3s[2,:], label='$\\alpha = 3$', color='tab:orange', marker='^', markersize=10, linestyle='--')
    
    axs[0].plot(Ns, l5s[2,:], label='$\\alpha = 3$', color='tab:blue', marker='s', markersize=10, linestyle='--')
    axs[1].plot(Ns, e5s[2,:], label='$\\alpha = 3$', color='tab:blue', marker='^', markersize=10, linestyle='--')

    axs2[0].set_title("c", fontsize='xx-large', fontweight='bold', loc='left')
    axs2[1].set_title("d", fontsize='xx-large', fontweight='bold', loc='left')
    axs[0].set_title("a", fontsize='xx-large', fontweight='bold', loc='left')
    axs[1].set_title("b", fontsize='xx-large', fontweight='bold', loc='left')
 """    
    for ax in axs:
        ax.tick_params(axis='both', labelsize='x-large')
        ax.set_xlabel("Number of Sites", fontsize='xx-large')
    
    custom_lines = [mpl.lines.Line2D([0], [0], color='k'), mpl.lines.Line2D([0], [0], color='k', ls='--')]
    axs[1].legend(custom_lines, ['Optimized', 'Uniform'], fontsize='x-large')

    axs[0].set_ylabel("Average Coherence Length", fontsize='xx-large')
    axs[1].set_ylabel("Average Population Flux", fontsize='xx-large')

#Keep
def all_resultenergy_pops_rhodiv_ramp(fnames=["results/ramp/usethese/12_01_10_neg10_ramppl22-01-2026_13:40.csv", "results/ramp/usethese/12_01_30_neg10_ramppl22-01-2026_13:33.csv", "results/ramp/usethese/12_01_50_neg10_ramppl22-01-2026_13:30.csv"], ufnames=["results/ramp/usethese/12_01_10_neg10_ramppl_singleG_22-01-2026_13:46.csv", "results/ramp/usethese/12_01_30_neg10_ramppl_singleG_22-01-2026_13:49.csv", "results/ramp/usethese/12_01_50_neg10_ramppl_singleG_22-01-2026_13:50.csv"], cmap='seismic', lognorm=True):
    fig, all_axs = plt.subplots(nrows=3, ncols=3, figsize=(16,16), layout='constrained', dpi=300)
    vmins = np.zeros(3)
    vmaxs = np.zeros(3)
    diffrhos = []
    for k in range(3):
        axs = all_axs[k,:]
        fname = fnames[k]
        #df = filterdf(fname)
        df = pd.read_csv(fname)
        df["eta"] = df["eta"].map(lambda x: np.abs(x))
        index = df["eta"].argmax()
        alpha = df["alpha"].iloc[index]
        if alpha <= 1:
            color = 'tab:purple'
        elif alpha <= 3:
            color = 'tab:orange'
        else:
            color = 'tab:blue'
        _, _, secax = energy_noise_ramp(fname, color2 = color, ax=axs[0])
        label = axs[0].set_ylabel(f"$\\alpha = {int(alpha)}$ \n Energy,", fontsize='xx-large')
        label = secax.set_ylabel("$\\Gamma$", color=color, fontsize='xx-large', rotation='vertical', horizontalalignment='right')
        N = df["N"].iloc[index]
        Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[index].values
        eps = df[["e" + str(k+1) for k in range(N)]].iloc[index].values
        ex = df["ex"].iloc[index]
        lk = df["lk"].iloc[index]
        gleak = df["gleak"].iloc[index]
        Jmax = df["Jmax"].iloc[index]

        rho = rhoss(jnp.array(Gammas), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        #dfu = filterdf(ufnames[k])
        dfu = pd.read_csv(ufnames[k])
        constG = dfu["Gamma"].iloc[np.argwhere(dfu["Unnamed: 0"].values == df["Unnamed: 0"].iloc[index]).item()]
        rho_u = rhoss(constG * jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        pops = jnp.abs(jnp.diagonal(rho, 0))
        pops_u = jnp.abs(jnp.diagonal(rho_u, 0))

        axs[1].plot(jnp.arange(1, N+1), pops_u, linewidth=0, marker='o', markersize=10, color='k', label=f'$\\Gamma_\\mathrm{{u}} = {round(constG, 4)}$')
        axs[1].plot(jnp.arange(1, N+1), pops, linewidth=0, marker='*', markersize=10, color=color, label='Optimized')
        axs[1].set_xlabel("Site", fontsize='xx-large')
        axs[1].set_ylabel("Population", fontsize='xx-large')
        axs[1].tick_params(axis='both', labelsize='x-large')
        axs[1].legend(fontsize='xx-large')

        axs[0].set_xticks(ticks=jnp.arange(2, N+2, 2), labels=jnp.arange(2, N+2, 2))
        axs[1].set_xticks(ticks=jnp.arange(2, N+2, 2), labels=jnp.arange(2, N+2, 2))
        
        if k < 2:
            for ax in axs:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelsize=0)

        diffrho = jnp.abs(rho) / jnp.abs(rho_u)
        diffrhos.append(diffrho)
        vmin = 10**jnp.min(-1.* jnp.abs(jnp.log10(diffrho)))
        vmax = 10**jnp.max(jnp.abs(jnp.log10(diffrho)))
        vmins[k] = vmin
        vmaxs[k] = vmax
    
    vmin = np.min(vmins)
    vmax = np.max(vmaxs)
    for k in range(3):
        axs = all_axs[k]
        diffrho = diffrhos[k]
        diffrho = jnp.fill_diagonal(diffrho, 1.0, inplace=False)
        """ if vmin > 0.1:
            vmin = 0.1
            vmax = 10. """
        if lognorm:
            im = axs[2].imshow(diffrho, cmap=cmap, norm=colors.SymLogNorm(linthresh = vmin.item()/10., linscale=1., vmin=vmin, vmax=vmax))
        else:
            im = axs[2].imshow(diffrho, cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=axs[2])
        
        axs[2].set_ylabel("Site", fontsize='xx-large')
        
            
        axs[2].set_xticks(ticks=range(0,N), labels=range(1,N+1))
        axs[2].set_yticks(ticks=range(0,N), labels=range(1,N+1))
        cbar.ax.set_ylabel("$|\\rho_o| / |\\rho_u|$", fontsize='xx-large')
        if k == 2:
            axs[2].tick_params(axis='both', labelsize='x-large')
            axs[2].set_xlabel("Site", fontsize='xx-large')
        else:
            axs[2].tick_params(axis='y', labelsize='x-large')
            axs[2].tick_params(axis='x', labelsize=0)
        cbar.ax.tick_params(axis='y', labelsize='x-large')


    alphabet = string.ascii_lowercase
    for k in range(all_axs.shape[0]):
        all_axs[k,0].set_title(alphabet[k] + "1", fontsize='xx-large', fontweight='bold', loc='left')
        all_axs[k,1].set_title(alphabet[k] + "2", fontsize='xx-large', fontweight='bold', loc='left')
        all_axs[k,2].set_title(alphabet[k] + "3", fontsize='xx-large', fontweight='bold', loc='left')
        
    
    return fig, all_axs

#Keep
def energy_noise_ramp(fname, color1='k', color2='r', ax=None, logscale=False):
    df = pd.read_csv(fname)
    if len(df) == 0:
        df = pd.read_csv(fname)
    if type(ax) == type(None):
        fig, ax = plt.subplots()
        axarg = False
    else:
        fig = ax.get_figure()
        axarg = True
    
    secax = ax.twinx()
    df["eta"] = df["eta"].map(lambda x: np.abs(x))
    index = df["eta"].argmax()
    N = df["N"].iloc[index]
    eps = df[["e" + str(k+1) for k in range(N)]].iloc[index].values
    Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[index].values

    

    #ax.plot(nums, eps, linewidth=0, marker='_', color=color1)
    generate_energy_level_diagram(eps, ax=ax, color=color1, linewidth=4)
    if logscale:
        secax.semilogy(range(1,N+1), Gammas, linewidth=0, marker='*', color=color2, markersize=10)
    else:
        secax.plot(range(1,N+1), Gammas, linewidth=0, marker='*', color=color2, markersize=10)

    ax.set_xlabel("Site", fontsize='xx-large')
    ax.set_ylabel("Site Energy", fontsize='xx-large', color=color1)
    #secax.set_ylabel("$\\Gamma$", fontsize='xx-large', color=color2, rotation='horizontal')

    ax.tick_params(axis='both', labelsize='x-large')
    secax.tick_params(axis='y', labelsize='x-large', labelcolor=color2)
    return fig, ax, secax
    """ if not axarg:
        return fig, ax """

#keep
def rhoplots_uniform_optimized_ramp(fnames, ufnames):
    alphabet = string.ascii_lowercase
    fig = plt.figure(figsize=(16,8), layout='constrained', dpi=300)

    gs = GridSpec(ncols=9, nrows=2, width_ratios=[8,0.5,1.5,8,0.5,1.5,8,0.5,1.5], height_ratios=[1,1]) #every third column empty for space
    axorhos = [fig.add_subplot(gs[0,3*k]) for k in range(3)] #cols mod 0

    axurhos = [fig.add_subplot(gs[1,3*k]) for k in range(3)] #cols mod 0

    
    caxes = [fig.add_subplot(gs[:2, 3*k+1]) for k in range(3)] #cols mod 1, colorbars for orho, urho 

    vmins1 = jnp.zeros(3)
    vmaxs1 = jnp.zeros(3)
    vmins2 = jnp.zeros(3)
    vmaxs2 = jnp.zeros(3)
    alphas = jnp.zeros(3)
    rhos = []
    urhos = []
    for j in range(3):
        fname = fnames[j]
        ufname = ufnames[j]
        df = pd.read_csv(fname)
        udf = pd.read_csv(ufname)
        df["eta"] = df["eta"].map(lambda x: np.abs(x))
        udf["eta"] = udf["eta"].map(lambda x: np.abs(x))
        real_ind = df["eta"].argmax()
        N = df["N"].iloc[real_ind]
        
        eps = df[["e" + str(k+1) for k in range(N)]].iloc[real_ind].values
        alpha = df["alpha"].iloc[real_ind]
        Jmax = df["Jmax"].iloc[real_ind]
        ex = df["ex"].iloc[real_ind]
        lk = df["lk"].iloc[real_ind]
        gleak = df["gleak"].iloc[real_ind]
        Gammas = df[["Gamma" + str(k+1) for k in range(N)]].iloc[real_ind].values
        
        Gamma = udf["Gamma"].iloc[np.argwhere(udf["Unnamed: 0"].values == df["Unnamed: 0"].iloc[real_ind]).item()]

        alphas = alphas.at[j].set(alpha)

        if alpha <= 1.:
            color = 'tab:purple'
            cmap = 'Purples'
        elif alpha <= 3.:
            color = 'tab:orange'
            cmap = 'Oranges'
        else:
            color = 'tab:blue'
            cmap = 'Blues'

        label = f"$\\alpha = {int(alpha)}$"

        rho = rhoss(jnp.array(Gammas), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        urho = rhoss(Gamma * jnp.ones(N), ex, lk, gleak, jnp.array(eps), Jmax, alpha)
        vmin = min(jnp.min(jnp.abs(rho)), jnp.min(jnp.abs(urho)))
        vmins1 = vmins1.at[j].set(vmin)

        pops = jnp.abs(jnp.diagonal(rho))
        upops = jnp.abs(jnp.diagonal(urho))

        rhodiv = jnp.abs(rho) / jnp.abs(urho)

        rho = jnp.fill_diagonal(jnp.abs(rho), val=0.0, inplace=False)
        urho = jnp.fill_diagonal(jnp.abs(urho), val=0.0, inplace=False)
        vmax = max(jnp.max(rho), jnp.max(urho))
        vmaxs1 = vmaxs1.at[j].set(vmax)

       
        
        
        
        vmins2 = vmins2.at[j].set(min(jnp.min(rhodiv), jnp.min(1/rhodiv)))
        vmaxs2 = vmaxs2.at[j].set(max(jnp.max(rhodiv), jnp.max(1/rhodiv)))

        rhos.append(rho)
        urhos.append(urho)



    vmin1 = jnp.min(vmins1)
    vmax1 = jnp.max(vmaxs1)

    vmin2 = jnp.min(vmins2)
    vmax2 = jnp.max(vmaxs2)

    for j in range(3):
        alpha = alphas[j]
        if alpha <= 1.:
            color = 'tab:purple'
            cmap = 'Purples'
        elif alpha <= 3.:
            color = 'tab:orange'
            cmap = 'Oranges'
        else:
            color = 'tab:blue'
            cmap = 'Blues'
        ax = axorhos[j]
        uax = axurhos[j]
        cax = caxes[j]

        rho = rhos[j]
        urho = urhos[j]


        im = ax.imshow(rho, cmap=cmap, norm='log', vmin=vmin1, vmax=vmax1, aspect='auto')
        uim = uax.imshow(urho, cmap=cmap, norm='log', vmin=vmin1, vmax=vmax1, aspect='auto')

        
        
        cbar = fig.colorbar(im, cax=cax)
        #cax.set_ylabel("$|\\rho|$", fontsize='xx-large', rotation='horizontal')
        cax.tick_params(axis='both', labelsize='x-large')
    
        
        
    #Axes loop
    for k in range(3):
        ax = axorhos[k]
        ax.tick_params(axis='both', labelsize=0)
        title = ax.set_title("$|\\rho_o|$", fontsize='xx-large')
        ax.annotate(f"$\\alpha = {int(alphas[k])}$\n", fontsize=25, xycoords=title, xy=(0.5,1), horizontalalignment='center', verticalalignment='bottom')
        ax.set_title(alphabet[k] + "1", fontsize='xx-large', fontweight='bold', loc='left')
        uax = axurhos[k]
        uax.tick_params(axis='both', labelsize=0)
        uax.set_title("$|\\rho_u|$", fontsize='xx-large')
        uax.set_title(alphabet[k] + "2", fontsize='xx-large', fontweight='bold', loc='left')
        
        
       
       
        if k > 0:
            ax.sharey(axorhos[0])
            uax.sharey(axurhos[0])
        


    for a in [axorhos[0], axurhos[0]]:
        a.set_yticks(range(1, N+1, 2), range(2, N+2, 2))
        a.set_ylabel("Site", fontsize='xx-large')
        a.tick_params(axis='y', labelsize='x-large')
    
    axurhos[1].set_xlabel("Site", fontsize='xx-large')
    for a in axurhos:
        a.set_xticks(range(1, N+1, 2), range(2, N+2, 2))
        a.tick_params(axis='x', labelsize='x-large')
    return fig, axurhos, caxes
