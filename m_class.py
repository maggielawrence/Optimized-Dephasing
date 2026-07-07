import jax
import jax.numpy as jnp
from typing import NamedTuple
import optax

#Keep
class Env(NamedTuple):
    N: int
    ex: int
    lk: int
    eps: jax.Array
    gleak: float
    Jmax: float
    alpha: float
    max_iter: int
    min_iter: int
    tol: float
    rate: float
    minGo: float
    maxGo: float
    exval: float
    
class Env2(NamedTuple):
    N: int
    H: jax.Array
    ex: int
    lk: int
    gleak: int
    max_iter: int
    min_iter: int
    tol: float
    rate: float
    minGo: float
    maxGo: float
    exval: float

class Optclasspl(NamedTuple):
    Gammas: jax.Array
    success: bool
    eta: float
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([jnp.where(self.Gammas >= env.minGo, True, False).all()])
        cond4 = jnp.array([jnp.where(self.Gammas <= env.maxGo, True, False).all()])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]

    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gammas)
        new_Gammas = optax.apply_updates(self.Gammas, updates)
        new_eta, new_grad = effgradfn(self.Gammas)
        new_count = self.count + 1
        return Optclasspl(Gammas=new_Gammas, count=new_count, eta=new_eta, grad=new_grad, success=self.success, opt_state=opt_state)
        
class Optclassalt(NamedTuple):
    Gammas: jax.Array
    success: bool
    eta: float
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([jnp.where(self.Gammas >= env.minGo, True, False).all()])
        cond4 = jnp.array([jnp.where(self.Gammas <= env.maxGo, True, False).all()])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]
    
    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gammas)
        new_Gammas = optax.apply_updates(self.Gammas, updates)
        new_eta, new_grad = effgradfn(self.Gammas)
        new_count = self.count + 1
        return Optclassalt(Gammas=new_Gammas, count=new_count, eta=new_eta, grad=new_grad, success=self.success, opt_state=opt_state)
    
#Keep
class Env1(NamedTuple):
    N: int
    ex: int
    lk: int
    eps: jax.Array
    gleak: float
    Jmax: float
    alpha: float
    max_iter: int
    min_iter: int
    tol: float
    rate: float
    minGo: float
    maxGo: float

class Optclasspl1(NamedTuple):
    Gamma: float
    success: bool
    eta: float
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([self.Gamma >= env.minGo])
        cond4 = jnp.array([self.Gamma <= env.maxGo])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]

    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gamma)
        new_Gamma = optax.apply_updates(self.Gamma, updates)
        new_eta, new_grad = effgradfn(self.Gamma)
        new_count = self.count + 1
        return Optclasspl1(Gamma=new_Gamma, count=new_count, eta=new_eta, grad=new_grad, success=self.success, opt_state=opt_state)

#Keep
class Optclasspl_dis(NamedTuple):
    Gammas: jax.Array
    success: bool
    eta: float
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([jnp.where(self.Gammas >= env.minGo, True, False).all()])
        cond4 = jnp.array([jnp.where(self.Gammas <= env.maxGo, True, False).all()])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]

    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gammas)
        new_Gammas = optax.apply_updates(self.Gammas, updates)
        new_eta, new_grad = effgradfn(self.Gammas, env.eps)
        new_count = self.count + 1
        return Optclasspl_dis(Gammas=new_Gammas, count=new_count, eta=new_eta, grad=new_grad, success=self.success, opt_state=opt_state)

class Optclasspl_traj(NamedTuple):
    Gammas: jax.Array
    all_Gammas: jax.Array
    success: bool
    eta: float
    all_etas: jax.Array
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([jnp.where(self.Gammas >= env.minGo, True, False).all()])
        cond4 = jnp.array([jnp.where(self.Gammas <= env.maxGo, True, False).all()])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]

    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gammas)
        new_Gammas = optax.apply_updates(self.Gammas, updates)
        new_eta, new_grad = effgradfn(self.Gammas)
        new_count = self.count + 1
        new_all_Gammas = self.all_Gammas.at[new_count,:].set(new_Gammas)
        new_all_etas = self.all_etas.at[new_count,:].set(new_eta)
        return Optclasspl_traj(Gammas=new_Gammas, all_Gammas=new_all_Gammas, count=new_count, eta=new_eta, all_etas=new_all_etas, grad=new_grad, success=self.success, opt_state=opt_state)

#Keep
class Optclasspl_dis1(NamedTuple):
    Gamma: float
    success: bool
    eta: float
    count: int
    grad: jax.Array
    opt_state: tuple

    def condfn(self, env):
        cond1 = jnp.array([jnp.where(jnp.abs(self.grad) > env.tol, True, False).any()])
        cond2 = jnp.array([self.count < env.max_iter])
        cond3 = jnp.array([self.Gamma >= env.minGo])
        cond4 = jnp.array([self.Gamma <= env.maxGo])
        cond = jnp.concatenate((cond1, cond2, cond3, cond4)).all()
        final = jnp.logical_or(jnp.array([self.count < env.min_iter]), cond)
        return final[0]

    def optimloop(self, effgradfn, solver, env):
        updates, opt_state = solver.update(self.grad, self.opt_state, self.Gamma)
        new_Gamma = optax.apply_updates(self.Gamma, updates)
        new_eta, new_grad = effgradfn(self.Gamma, env.eps)
        new_count = self.count + 1
        return Optclasspl_dis1(Gamma=new_Gamma, count=new_count, eta=new_eta, grad=new_grad, success=self.success, opt_state=opt_state)


