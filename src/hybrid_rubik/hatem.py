#!/usr/bin/env python3
from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

EPS = 1e-9

def normalize_progress(best_trace: List[float]) -> np.ndarray:
    tr = np.array(best_trace, dtype=float)
    O0 = float(tr[0]); Omin = float(tr.min())
    P = (O0 - tr) / ((O0 - Omin) + EPS)
    return np.clip(P, 0.0, 1.0)

def make_target_base(n: int, kind: str, params: dict) -> np.ndarray:
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    if kind == "trefoil":
        a,b,c = params.get("a",1.0), params.get("b",2.0), params.get("c",1.0)
        x = a*np.sin(t) + b*np.sin(2*t); y = a*np.cos(t) - b*np.cos(2*t); z = -c*np.sin(3*t)
    elif kind == "circle":
        r,z0 = params.get("r",1.0), params.get("z0",0.0); x = r*np.cos(t); y = r*np.sin(t); z = z0 + 0*t
    elif kind == "lissajous":
        ax,ay,az = params.get("ax",3), params.get("ay",4), params.get("az",5)
        phx,phy,phz = params.get("phx",0.0), params.get("phy",0.6), params.get("phz",0.2)
        sx,sy,sz = params.get("sx",1.0), params.get("sy",1.0), params.get("sz",1.0)
        x = sx*np.sin(ax*t + phx); y = sy*np.sin(ay*t + phy); z = sz*np.sin(az*t + phz)
    else:
        p,q,r = params.get("p",2), params.get("q",3), params.get("r",4)
        sx,sy,sz = params.get("sx",1.0), params.get("sy",1.0), params.get("sz",1.0)
        x = sx*np.cos(p*t); y = sy*np.sin(q*t); z = sz*np.sin(r*t)
    return np.stack([x,y,z], axis=1)

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(3,3)); Q,_ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

def make_target(n: int, kind: str, seed: int = 0, params: Optional[dict]=None) -> np.ndarray:
    rng = np.random.default_rng(seed); params = {} if params is None else dict(params)
    if not params:
        if kind == "trefoil": params = {"a": rng.uniform(0.7,1.3), "b": rng.uniform(1.3,2.7), "c": rng.uniform(0.7,1.4)}
        elif kind == "circle": params = {"r": rng.uniform(0.8,1.4), "z0": rng.uniform(-0.2,0.2)}
        elif kind == "lissajous":
            params = {"ax": int(rng.integers(2,6)), "ay": int(rng.integers(3,7)), "az": int(rng.integers(4,8)),
                      "phx": rng.uniform(0,2*np.pi), "phy": rng.uniform(0,2*np.pi), "phz": rng.uniform(0,2*np.pi),
                      "sx": rng.uniform(0.7,1.4), "sy": rng.uniform(0.7,1.4), "sz": rng.uniform(0.7,1.4)}
        else:
            params = {"p": int(rng.integers(1,5)), "q": int(rng.integers(2,6)), "r": int(rng.integers(3,7)),
                      "sx": rng.uniform(0.7,1.4), "sy": rng.uniform(0.7,1.4), "sz": rng.uniform(0.7,1.4)}
    X = make_target_base(n, kind, params); R = random_rotation(rng); scale = rng.uniform(0.8,1.6); shift = rng.uniform(-0.4,0.4, size=(1,3))
    return (X @ R) * scale + shift

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P - P.mean(axis=0, keepdims=True); Qc = Q - Q.mean(axis=0, keepdims=True); C = Pc.T @ Qc
    V,_,Wt = np.linalg.svd(C); d = np.sign(np.linalg.det(V @ Wt)); R = V @ np.diag([1.0,1.0,d]) @ Wt
    t = Q.mean(axis=0) - (P.mean(axis=0) @ R); return R,t

def min_dist_sample(X: np.ndarray) -> float:
    n = X.shape[0]; i = np.arange(n); j = (i+9) % n
    return float(np.linalg.norm(X[i]-X[j], axis=1).min())

SCENARIOS: Dict[str, Dict[str, float]] = {
    "baseline": dict(bloat=0.0, heat=0.0, electric=0.0), "bloat": dict(bloat=+0.25, heat=0.0, electric=0.0),
    "shrink": dict(bloat=-0.20, heat=0.0, electric=0.0), "electric": dict(bloat=0.0, heat=0.0, electric=+1.0),
    "heat_add": dict(bloat=0.0, heat=+1.0, electric=0.0), "heat_remove": dict(bloat=0.0, heat=-1.0, electric=0.0),
}
CURRICULUM = [("baseline",0.45),("heat_remove",0.25),("heat_add",0.20),("bloat",0.05),("shrink",0.03),("electric",0.02)]

def sample_curriculum_scenario(rng: np.random.Generator, stage: float) -> str:
    names=[k for k,_ in CURRICULUM]; w0=np.array([w for _,w in CURRICULUM],float); w0=w0/w0.sum(); wu=np.ones_like(w0)/len(w0); w=(1-stage)*w0+stage*wu; w=w/w.sum(); return str(rng.choice(names,p=w))

def sample_target_family(rng: np.random.Generator, stage: float) -> str:
    fams=["circle","trefoil","alt","lissajous"]; w0=np.array([0.45,0.35,0.15,0.05],float); wu=np.ones_like(w0)/len(w0); w=(1-stage)*w0+stage*wu; w=w/w.sum(); return str(rng.choice(fams,p=w))

class KnotEnergy:
    def __init__(self,n:int,seed:int=0):
        rng=np.random.default_rng(seed); self.n=n; self.freq=np.linspace(0.5,2.5,n)+rng.normal(0,0.05,n); self.vib=rng.uniform(0.8,1.4,n); self.e0=rng.uniform(0.5,1.5,n); self.maxfreq=float(self.freq.max()); self.eval_count=0; self.i=np.arange(n); self.j=(self.i+7)%n; self.rep_w=(0.7+0.6*(self.freq[self.i]+self.freq[self.j])/self.maxfreq)
    def energy(self,X:np.ndarray,Xt:np.ndarray,l0:float,scen:Dict[str,float],w_rep:float)->float:
        self.eval_count+=1; Xn=np.roll(X,-1,axis=0); Xm=np.roll(X,1,axis=0); seg=Xn-X; seg_len=np.linalg.norm(seg,axis=1)+1e-12
        k_i=(self.e0*self.vib)*(0.8+0.6*(self.freq/np.mean(self.freq))); l_pref=l0*(1.0+scen["bloat"]); E_stretch=float(np.sum(k_i*(seg_len-l_pref)**2))
        d2=Xn-2*X+Xm; E_bend=float(np.sum((0.3+0.7*self.vib)*(np.linalg.norm(d2,axis=1)**2)))
        rij=X[self.i]-X[self.j]; r2=np.sum(rij*rij,axis=1)+1e-3; E_rep=float(np.sum(self.rep_w/(r2*r2)))
        R,t=kabsch(X,Xt); Xal=X@R+t; E_target=float(np.sum((Xal-Xt)**2)); E_elec=float(np.sum(scen["electric"]*self.freq*X[:,0])) if scen["electric"]!=0.0 else 0.0
        return 3.0*E_stretch+0.6*E_bend+w_rep*E_rep+E_target+0.15*E_elec

class TinyPolicyNet:
    def __init__(self,in_dim:int,hid:int=14,seed:int=0):
        rng=np.random.default_rng(seed); self.W1=rng.normal(scale=0.2,size=(in_dim,hid)); self.b1=np.zeros(hid); self.W2=rng.normal(scale=0.2,size=(hid,1)); self.b2=np.zeros(1)
    def forward(self,x:np.ndarray):
        h=np.tanh(x@self.W1+self.b1); y=h@self.W2+self.b2; return float(y.squeeze()),h
    def update(self,x:np.ndarray,target:float,lr:float=0.03):
        y,h=self.forward(x); d=2.0*(y-target); dW2=d*h[:,None]; dh=(d*self.W2.squeeze())*(1.0-h*h); dW1=x[:,None]@dh[None,:]; self.W2-=lr*dW2; self.b2-=lr*d; self.W1-=lr*dW1; self.b1-=lr*dh
    def to_dict(self)->dict: return {"W1":self.W1.tolist(),"b1":self.b1.tolist(),"W2":self.W2.tolist(),"b2":self.b2.tolist()}
    @staticmethod
    def from_dict(d:dict)->"TinyPolicyNet":
        W1=np.array(d["W1"],float); net=TinyPolicyNet(W1.shape[0],W1.shape[1],0); net.W1=W1; net.b1=np.array(d["b1"],float); net.W2=np.array(d["W2"],float); net.b2=np.array(d["b2"],float); return net
    def save(self,path:str)->None:
        with open(path,"w") as f: json.dump(self.to_dict(),f)
    @staticmethod
    def load(path:str)->"TinyPolicyNet":
        with open(path) as f: return TinyPolicyNet.from_dict(json.load(f))

class UCB:
    def __init__(self,names,c:float=1.4): self.names=list(names); self.c=c; self.t=0; self.n={k:0 for k in self.names}; self.q={k:0.0 for k in self.names}
    def pick(self)->str:
        self.t+=1
        for k in self.names:
            if self.n[k]==0: return k
        vals=[self.q[k]+self.c*math.sqrt(math.log(self.t+1)/(self.n[k]+1e-9)) for k in self.names]; return self.names[int(np.argmax(vals))]
    def update(self,k:str,r:float,alpha:float=0.15): self.n[k]+=1; self.q[k]=(1-alpha)*self.q[k]+alpha*r

@dataclass
class SolveResult:
    solver:str; time_s:float; evals:int; auc:float; score:float; best_trace:List[float]; eval_trace:List[int]

def hybrid_adaptive(n:int,scen:Dict[str,float],seed:int=7,budget:int=2000,kind:str="trefoil",target_seed:int=0,policy:Optional[TinyPolicyNet]=None,learn:bool=True):
    rng=np.random.default_rng(seed); Xt=make_target(n,kind,seed=target_seed); ke=KnotEnergy(n,seed); X=Xt+rng.normal(0,0.8,Xt.shape); l0=float(np.mean(np.linalg.norm(np.roll(Xt,-1,axis=0)-Xt,axis=1)))
    heat=scen["heat"]; T0=0.7*(1+0.9*max(0.0,heat))*(1-0.4*max(0.0,-heat)); w_rep=0.22; arms=["swap2","swap3","swap4","rot3","rot4"]; ucb=UCB(arms,1.4); in_dim=11; net=policy if policy is not None else TinyPolicyNet(in_dim,14,seed+99)
    E=ke.energy(X,Xt,l0,scen,w_rep); best=E; best_trace=[best]; eval_trace=[ke.eval_count]; it=0; t0=time.perf_counter()
    while ke.eval_count+5<=budget:
        it+=1; curv=0.0
        if it%8==0 and ke.eval_count+2<=budget:
            delta=rng.choice([-1.0,1.0],size=X.shape); Ep=ke.energy(X+0.02*delta,Xt,l0,scen,w_rep); Em=ke.energy(X-0.02*delta,Xt,l0,scen,w_rep); curv=abs(Ep-Em)/0.04
        macro_every=max(3,7-int(min(4,curv/60.0)))
        if it%macro_every==0 and ke.eval_count+1<=budget:
            arm=ucb.pick(); L=int(arm[-1]); bestcand=None
            for _ in range(6):
                a=int(rng.integers(0,n)); ia=[(a+i)%n for i in range(L)]; Xp=X.copy()
                if arm.startswith("swap"):
                    b=int(rng.integers(0,n)); ib=[(b+i)%n for i in range(L)]; tmp=Xp[ia].copy(); Xp[ia]=Xp[ib]; Xp[ib]=tmp; mismatch=abs(float(np.mean(ke.freq[ia])-np.mean(ke.freq[ib])))
                else:
                    dir_=1 if rng.random()<0.5 else -1; block=Xp[ia].copy(); block=np.concatenate([block[-1:],block[:-1]],axis=0) if dir_==1 else np.concatenate([block[1:],block[:1]],axis=0); Xp[ia]=block; mismatch=float(np.std(ke.freq[ia]))
                dmin=min_dist_sample(Xp)
                if dmin<0.14: continue
                xfeat=np.zeros(in_dim,float); xfeat[0]=L/4.0; xfeat[1]=mismatch; xfeat[2]=dmin; xfeat[3]=heat; xfeat[4]=scen["electric"]; xfeat[5]=min(1.0,curv/240.0); xfeat[6+arms.index(arm)]=1.0
                pred,_=net.forward(xfeat); sc=pred-0.12*mismatch+0.08*dmin
                if bestcand is None or sc>bestcand[0]: bestcand=(sc,xfeat,Xp,arm)
            if bestcand is not None:
                _,xfeat,Xm,arm=bestcand; Em=ke.energy(Xm,Xt,l0,scen,w_rep); dE=Em-E; T=max(1e-6,T0/(1+it/40))
                if dE<=0 or random.random()<math.exp(-dE/T): X=Xm; E=Em; best=min(best,E)
                reward=max(0.0,best_trace[-1]-best)
                if learn: net.update(xfeat,reward,0.03)
                ucb.update(arm,reward,0.15)
        if ke.eval_count+3>budget: break
        a=0.03/(it**0.15); eta=0.010/(it**0.2); eta*=(1+0.35*max(0.0,heat))*(1-0.2*max(0.0,-heat)); delta=rng.choice([-1.0,1.0],size=X.shape)
        Ep=ke.energy(X+a*delta,Xt,l0,scen,w_rep); Em=ke.energy(X-a*delta,Xt,l0,scen,w_rep); ghat=(Ep-Em)/(2*a)*delta; sigma=0.014*(1+0.8*max(0.0,heat))*(1-0.5*max(0.0,-heat)); Xp=X-eta*ghat+rng.normal(scale=max(0.0,sigma),size=X.shape); Xp-=Xp.mean(axis=0,keepdims=True)
        Eprop=ke.energy(Xp,Xt,l0,scen,w_rep); dE=Eprop-E; T=max(1e-6,T0/(1+it/40))
        if dE<=0 or random.random()<math.exp(-dE/T): X=Xp; E=Eprop; best=min(best,E)
        best_trace.append(best); eval_trace.append(ke.eval_count)
    P=normalize_progress(best_trace); auc=float(P.mean()); score=float(0.7*P[-1]+0.3*auc); return SolveResult("Hybrid-Adaptive",time.perf_counter()-t0,int(ke.eval_count),auc,score,best_trace,eval_trace),net

def knot_continuous(n:int,scen:Dict[str,float],seed:int=7,budget:int=2000,kind:str="trefoil",target_seed:int=0)->SolveResult:
    rng=np.random.default_rng(seed); Xt=make_target(n,kind,seed=target_seed); ke=KnotEnergy(n,seed); X=Xt+rng.normal(0,0.65,Xt.shape); l0=float(np.mean(np.linalg.norm(np.roll(Xt,-1,axis=0)-Xt,axis=1))); heat=scen["heat"]; T0=0.6*(1+0.9*max(0.0,heat))*(1-0.4*max(0.0,-heat)); w_rep=0.22
    E=ke.energy(X,Xt,l0,scen,w_rep); best=E; best_trace=[best]; eval_trace=[ke.eval_count]; it=0; t0=time.perf_counter()
    while ke.eval_count+3<=budget:
        it+=1; a=0.03/(it**0.15); eta=0.012/(it**0.2); eta*=(1+0.35*max(0.0,heat))*(1-0.2*max(0.0,-heat)); delta=rng.choice([-1.0,1.0],size=X.shape); Ep=ke.energy(X+a*delta,Xt,l0,scen,w_rep); Em=ke.energy(X-a*delta,Xt,l0,scen,w_rep); ghat=(Ep-Em)/(2*a)*delta; sigma=0.02*(1+0.8*max(0.0,heat))*(1-0.5*max(0.0,-heat)); Xp=X-eta*ghat+rng.normal(scale=max(0.0,sigma),size=X.shape); Xp-=Xp.mean(axis=0,keepdims=True); Eprop=ke.energy(Xp,Xt,l0,scen,w_rep); dE=Eprop-E; T=max(1e-6,T0/(1+it/40))
        if dE<=0 or random.random()<math.exp(-dE/T): X=Xp; E=Eprop; best=min(best,E)
        best_trace.append(best); eval_trace.append(ke.eval_count)
    P=normalize_progress(best_trace); auc=float(P.mean()); score=float(0.7*P[-1]+0.3*auc); return SolveResult("Knot-Continuous",time.perf_counter()-t0,int(ke.eval_count),auc,score,best_trace,eval_trace)
