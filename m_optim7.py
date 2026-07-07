import sys
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd
import os
from datetime import datetime
import optax
import argparse
from m_class import Env, Env1, Optclasspl_dis, Optclasspl_dis1
#'Smart' starting conditions for optimizer
jax.config.update('jax_enable_x64', True)

def eff(Gammas, ex, lk, gleak, eps, Jmax, alpha):
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

    prob = jnp.abs(rho[lk,lk])

    return gleak * prob

def eff1(Gamma, ex, lk, gleak, eps, Jmax, alpha):
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

    klambda = lambda k: Gamma * kloop(k, N)
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

    prob = jnp.abs(rho[lk,lk])

    return gleak * prob


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

def producer1(Gamma, effgradfn, solver, env):
    opt_state = solver.init(Gamma)
    success = True
    Gamma0 = Gamma
    (eta, grad) = effgradfn(Gamma, env.eps)
    count = 0
    eta0 = eta
    opt = Optclasspl_dis1(Gamma=Gamma, success=success, eta=eta, count=count, grad=grad, opt_state=opt_state)

    #Fill in opt for x here
    optimlambda = lambda x: x.optimloop(effgradfn, solver, env)
    condlambda = lambda x: x.condfn(env)
    opt = jax.lax.while_loop(
        condlambda,
        optimlambda,
        opt
    )
    opt = jax.lax.cond(jnp.astype(opt.count, float) >= env.max_iter, lambda opt: Optclasspl_dis1(Gamma = opt.Gamma, success=False, eta=opt.eta, count=opt.count, grad=opt.grad, opt_state=opt.opt_state), lambda opt: opt, opt)
    concat = jnp.concatenate((jnp.array([Gamma0]), jnp.array([opt.Gamma]), jnp.array([eta0]), jnp.array([opt.eta]), jnp.array([opt.count]), jnp.array([opt.success])))
    return concat

def producer(Gammas, effgradfn, solver, env):
    opt_state = solver.init(Gammas)
    success = True
    Gammas0 = Gammas
    (eta, grad) = effgradfn(Gammas, env.eps)
    count = 0
    eta0 = eta
    opt = Optclasspl_dis(Gammas=Gammas, success=success, eta=eta, count=count, grad=grad, opt_state=opt_state)

    #Fill in opt for x here
    optimlambda = lambda x: x.optimloop(effgradfn, solver, env)
    condlambda = lambda x: x.condfn(env)
    opt = jax.lax.while_loop(
        condlambda,
        optimlambda,
        opt
    )
    opt = jax.lax.cond(jnp.astype(opt.count, float) >= env.max_iter, lambda opt: Optclasspl_dis(Gammas = opt.Gammas, success=False, eta=opt.eta, count=opt.count, grad=opt.grad, opt_state=opt.opt_state), lambda opt: opt, opt)
    opt = jax.lax.cond(jnp.array([jnp.where(jnp.abs(opt.grad) > env.tol, True, False).any()])[0], lambda opt: Optclasspl_dis(Gammas = opt.Gammas, success = False, eta = opt.eta, count = opt.count, grad = opt.grad, opt_state=opt_state), lambda opt: opt, opt)
    if len(Gammas0) == env.N:
        concat = jnp.concatenate((Gammas0, opt.Gammas, jnp.array([eta0]), jnp.array([opt.eta]), jnp.array([opt.count]), jnp.array([opt.success])))
    else:
        concat = jnp.concatenate((jnp.array([env.exval]), Gammas0, jnp.array([env.exval]), opt.Gammas, jnp.array([eta0]), jnp.array([opt.eta]), jnp.array([opt.count]), jnp.array([opt.success])))
    return concat


def sample_nd_grid(ndim, lower_bounds, upper_bounds, num_points, num_samples):
    if len([lower_bounds]) == 1:
        lower_bounds = lower_bounds * jnp.ones(ndim)
    if len([upper_bounds]) == 1:
        upper_bounds = upper_bounds * jnp.ones(ndim)
    if len([num_points]) == 1:
        num_points = num_points * jnp.ones(ndim, dtype=int)
    # Create the N-dimensional grid
    grid = jnp.array(jnp.meshgrid(*[jnp.linspace(lower, upper, num) for lower, upper, num in zip(lower_bounds, upper_bounds, num_points)]))
    
    # Reshape the grid to have each point in a separate row
    points = grid.reshape(ndim, -1).T
    
    # Sample without replacement
    indices = np.random.choice(points.shape[0], size=num_samples, replace=False)
    sampled_points = points[indices]
    
    # Convert the points to a list of tuples
    #sampled_points = list(map(tuple, sampled_points))
    
    return sampled_points

def no_decimal_string(number):
    # Convert number to string
    number_str = str(number)
    if number <0:
        neg_flag = "neg"
        number_str = number_str.lstrip("-")
    else: neg_flag = "" 
    
    # Check if the number has a decimal point
    if '.' in number_str:
        # Split the number into integer and decimal parts
        integer_part, decimal_part = number_str.split('.')
        
        # If the decimal part is empty, return the integer part
        if len(decimal_part) == 0:
            return neg_flag + integer_part
        else:
            return neg_flag + integer_part + decimal_part
        
    else:
        # If the number doesn't have a decimal point, return it as is
        return neg_flag + number_str
    

def main():
    
    start = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--Num", type=int, default=3, help="Number of sites")
    parser.add_argument("--Jmax", type=float, default=0.2, help="Jmax")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--gleak", type=float, default=0.1, help="Leak rate")
    parser.add_argument("--Delta", type=float, default=1., help="Maximum energy difference, not used for uniform profile")
    parser.add_argument("--cluster", type=int, default=0, help="Set to 1 if doing calculations on the cluster (default 0)")
    parser.add_argument("--save", type=int, default=1, help="Set to 1 to save the optimization results (default 1)")
    parser.add_argument("--points", type=int, default=4, help="Points per dimension to use when generating hypergrid of possible initial energy profiles")
    parser.add_argument("--samples", type=int, default=1, help="How many initial energy configurations to sample from hypergrid and optimize")
    parser.add_argument("--rate", type=float, default = 0.01, help="Learning rate for optimizer")
    parser.add_argument("--minGg", type=float, default = 1e-7, help="Minimum Gamma for hypergrid")
    parser.add_argument("--maxGg", type=float, default=1., help="Maximum Gamma for hypergrid")
    parser.add_argument("--minGo", type=float, default=1e-7, help="Minimum Gamma for optimizer")
    parser.add_argument("--maxGo", type=float, default=1., help="Maximum Gamma for optimizer")
    parser.add_argument("--ex", type=int, default=0, help="Injection site index")
    parser.add_argument("--lk", type=int, default=2, help="Leak site index")
    parser.add_argument("--maxiter", type=int, default=100000, help="Max number of steps to do per optimization")
    parser.add_argument("--miniter", type=int, default=30, help="Min number of steps to do per optimization")
    parser.add_argument("--tol", type=float, default=1e-4, help="Stopping condition")
    parser.add_argument("--optim", type=str, default="adam", help="Optax solver")
    parser.add_argument("--exfix", type=int, default=0, help="Whether to force the excited site's noise to be fixed")
    parser.add_argument("--exval", type=float, default=0.0, help="Noise value on excited site if exfix is True, default 0.0")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = parser.parse_args()
    rng = np.random.default_rng(seed=args.seed)
    N = args.Num
    minGg = args.minGg
    maxGg = args.maxGg
    minGo = args.minGo
    maxGo = args.maxGo
    points_per_dim = args.points
    samples = args.samples
    ex = args.ex
    if args.lk == -1:
        lk = N - 1
    else:
        lk = args.lk
    gleak = args.gleak
    Jmax = args.Jmax
    alpha = args.alpha
    max_iter = args.maxiter
    min_iter = args.miniter
    tol = args.tol
    rate = args.rate
    eps = jnp.array(args.Delta * rng.random(size=(samples, N)))
    profile = "disordered"
    mystring = "global solver; solver = optax."+ args.optim + "(learning_rate=" + str(args.rate) + ")"
    exec(mystring)
    
    ## Find best uniform Gamma
    env1 = Env1(N=N, ex=ex, lk=lk, eps=eps, gleak=gleak, Jmax=Jmax, alpha=alpha, max_iter=max_iter, min_iter=min_iter, tol=tol, rate=rate, minGo=minGo, maxGo=maxGo)
    powers1 = jnp.linspace(jnp.log10(minGg), jnp.log10(maxGg), samples)
    f1 = lambda G, e: -1.*eff1(10.0**G, ex, lk, gleak, e, Jmax, alpha)
    f = lambda G, e: -1.*eff(10.0**G, ex, lk, gleak, e, Jmax, alpha)
    effgradfn1 = jax.value_and_grad(f1, argnums=(0))
    effgradfn = jax.value_and_grad(f, argnums=(0))
    prodvec1 = jax.vmap(producer1, in_axes=(0, None, None, Env1(None, None, None, 0, None, None, None, None, None, None, None, None, None)))
    prodvec = jax.vmap(producer, in_axes=(0, None, None, Env(None, None, None, 0, None, None, None, None, None, None, None, None, None, None)))

    opt_powers1 = jnp.zeros(samples)
    for j in range(samples):
        powers1 = jnp.linspace(jnp.log10(minGo), jnp.log10(maxGo), samples)
        effs = jnp.array([f1(powers1[k], eps[j]) for k in range(len(powers1))])
        opt_powers1 = opt_powers1.at[j].set(powers1[jnp.argmin(effs)])
    
    ## Multivariable optimization
    opt_powers = jnp.multiply(jnp.tile(opt_powers1.reshape((-1,1)), N), jnp.ones((samples, N)))

    env = Env(N=N, ex=ex, lk=lk, eps=eps, gleak=gleak, Jmax=Jmax, alpha=alpha, max_iter=max_iter, min_iter=min_iter, tol=tol, rate=rate, minGo=jnp.log10(minGo), maxGo=jnp.log10(maxGo), exval=args.exval)
    
    ## producer(x, effgradfn, solver, env)
    prodvec = jax.vmap(producer, in_axes=(0, None, None, Env(None, None, None, 0, None, None, None, None, None, None, None, None, None, None)))

    result = prodvec(opt_powers, effgradfn, solver, env)

    gnames0 = ["Gamma0_" + str(k+1) for k in range(N)]
    gnames = ["Gamma" + str(k+1) for k in range(N)]
    colnames = gnames0 + gnames + ["eta0", "eta", "count", "success"]

    array = np.array(result)

    df = pd.DataFrame(array, columns=colnames)

    if args.exfix == 0:
        df[gnames0] = df[gnames0].map(lambda x: 10.**x)
        df[gnames] = df[gnames].map(lambda x: 10.**x)
    else:
        df[gnames0[1:]] = df[gnames0[1:]].map(lambda x: 10.**x)
        df[gnames[1:]] = df[gnames[1:]].map(lambda x: 10.**x)

    for k in range(N):
        df.insert(len(df.columns), "e" + str(k+1), eps[:,k])
    df.insert(len(df.columns), "max_iter", max_iter)
    df.insert(len(df.columns), "min_iter", min_iter)
    df.insert(len(df.columns), "N", N)
    df.insert(len(df.columns), "Jmax", Jmax)
    df.insert(len(df.columns), "alpha", alpha)
    df.insert(len(df.columns), "gleak", gleak)
    df.insert(len(df.columns), "ex", ex)
    df.insert(len(df.columns), "lk", lk)
    df.insert(len(df.columns), "rate", rate)
    df.insert(len(df.columns), "minGg", minGg)
    df.insert(len(df.columns), "maxGg", maxGg)
    df.insert(len(df.columns), "minGo", minGo)
    df.insert(len(df.columns), "maxGo", maxGo)
    df.insert(len(df.columns), "tol", tol)
    df.insert(len(df.columns), "optim", args.optim)
    df.insert(len(df.columns), "exfix", args.exfix)
    df.insert(len(df.columns), "exval", args.exval)
    df.insert(len(df.columns), "profile", profile)
    df.insert(len(df.columns), "seed", args.seed)
    df.insert(len(df.columns), "Gamma", opt_powers1)
    df["Gamma"] = df["Gamma"].map(lambda x: 10.**x)

    #df.sort_values(by = "eta", ascending=True, inplace=True)
    print(df)

    if args.exfix == 0:
        job_name = str(N) + "_" + no_decimal_string(Jmax) + "_" + no_decimal_string(alpha) + "_" + no_decimal_string(args.Delta) + "_" + profile + "pl" + "_seed_" + str(args.seed) + "_smart_"
    else:
        job_name = str(N) + "_" + no_decimal_string(Jmax) + "_" + no_decimal_string(alpha) + "_" + profile + "pl_exfix_"

    now = datetime.now()
    df.insert(len(df.columns), "runtime", now-start)

    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    if args.save == True:
        if args.cluster == True:
            filepath = os.path.join(os.path.expanduser('~'), os.getcwd(), "elise_results", job_name + dt_string + ".csv")
        else:
            filepath = os.path.join(os.path.expanduser('~'), os.getcwd(), "results", profile, job_name + dt_string + ".csv")
        df.to_csv(filepath)

    return 0

if __name__ == "__main__":
    main()


