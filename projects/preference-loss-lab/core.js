"use strict";
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const softplus=x=>Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
function logits(w,x){return w.map(row=>row.reduce((s,v,i)=>s+v*x[i],0));}
function probabilities(w,x){const z=logits(w,x),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function validate(w,ref,pairs,beta){
 if(!Number.isFinite(beta)||beta<=0||!Array.isArray(w)||w.length<2||!w[0].length||w.length!==ref.length||!pairs.length)throw Error("Invalid DPO configuration");
 const d=w[0].length;
 if(![...w,...ref].every(row=>row.length===d&&row.every(Number.isFinite)) ||
 !pairs.every(p=>p.x.length===d&&p.x.every(Number.isFinite)&&Number.isInteger(p.chosen)&&Number.isInteger(p.rejected)&&p.chosen>=0&&p.rejected>=0&&p.chosen<w.length&&p.rejected<w.length&&p.chosen!==p.rejected))throw Error("Invalid weights or preferences");
}
function dpoLossGrad(w,ref,pairs,beta=.5){
 validate(w,ref,pairs,beta);
 const grad=w.map(row=>row.map(()=>0));let loss=0;
 for(const p of pairs){
  const z=logits(w,p.x),r=logits(ref,p.x);
  const margin=(z[p.chosen]-z[p.rejected])-(r[p.chosen]-r[p.rejected]);
  loss+=softplus(-beta*margin);
  const g=-beta*sigmoid(-beta*margin);
  for(let j=0;j<p.x.length;j++){grad[p.chosen][j]+=g*p.x[j]/pairs.length;grad[p.rejected][j]-=g*p.x[j]/pairs.length;}
 }
 return {loss:loss/pairs.length,grad};
}
function sftLossGrad(w,pairs){
 const grad=w.map(row=>row.map(()=>0));let loss=0;
 for(const p of pairs){
  const z=logits(w,p.x),m=Math.max(...z),logsum=m+Math.log(z.reduce((s,v)=>s+Math.exp(v-m),0));
  loss+=logsum-z[p.chosen];const prob=probabilities(w,p.x);
  for(let a=0;a<w.length;a++)for(let j=0;j<p.x.length;j++)grad[a][j]+=(prob[a]-(a===p.chosen?1:0))*p.x[j]/pairs.length;
 }
 return {loss:loss/pairs.length,grad};
}
function evaluate(w,ref,pairs){
 let correct=0,kl=0;
 for(const p of pairs){
  const z=logits(w,p.x),prob=probabilities(w,p.x),r=probabilities(ref,p.x);
  correct+=z[p.chosen]>z[p.rejected]?1:0;
  kl+=prob.reduce((s,v,i)=>s+(v===0?0:v*(Math.log(v)-Math.log(r[i]))),0);
 }
 return {pairAccuracy:correct/pairs.length,meanKL:kl/pairs.length};
}
module.exports={dpoLossGrad,sftLossGrad,evaluate,logits,probabilities};
