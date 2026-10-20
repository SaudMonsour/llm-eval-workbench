"use strict";
const {dpoLossGrad,sftLossGrad,evaluate}=require("./core");
function rng(seed){return()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};}
function fixture(count,seed){
 const r=rng(seed),truth=[[1.1,-.8,.4,.2,-.5,.3],[-.3,1.2,-.7,.5,.4,-.1],[.2,-.3,1.1,-.8,.6,-.2]];
 return Array.from({length:count},()=>{
  const x=Array.from({length:6},()=>2*r()-1),a=Math.floor(r()*3),b=(a+1+Math.floor(r()*2))%3;
  const score=i=>truth[i].reduce((s,v,j)=>s+v*x[j],0);
  return {x,chosen:score(a)>score(b)?a:b,rejected:score(a)>score(b)?b:a};
 });
}
function run(){
 const train=fixture(600,41),validation=fixture(200,42),holdout=fixture(200,43);
 const reference=[[.05,-.03,.01,0,.02,-.01],[-.01,.01,.02,-.04,.01,.03],[.02,0,-.01,.02,-.02,.01]];
 const rows=[{method:"Frozen reference",...evaluate(reference,reference,holdout)}],curves={},checkpoints={};
 for(const [name,beta] of [["SFT",null],["DPO beta 0.1",.1],["DPO beta 0.5",.5],["DPO beta 1",1]]){
  let w=reference.map(row=>row.slice()),best=Infinity,saved=null,selectedStep=0;const curve=[];
  const objective=p=>beta===null?sftLossGrad(w,p):dpoLossGrad(w,reference,p,beta);
  for(let step=1;step<=400;step++){
   const g=objective(train).grad;
   for(let a=0;a<3;a++)for(let j=0;j<6;j++)w[a][j]-=.2*g[a][j];
   const val=objective(validation).loss;curve.push(val);
   if(val<best){best=val;saved=w.map(row=>row.slice());selectedStep=step;}
  }
  curves[name]=curve;checkpoints[name]=saved;
  rows.push({method:name,selectedStep,validationObjective:best,...evaluate(saved,reference,holdout)});
 }
 return {seedPolicy:"Fixed train=41, validation=42, holdout=43; no seed search",fixture:"Synthetic six-feature contexts, three candidate actions; deterministic linear preference oracle; not language-model responses",train:600,validation:200,holdout:200,optimizer:"Full-batch SGD, lr=0.2, 400 steps; minimum per-method validation objective checkpoint",results:rows,curves,checkpoints,reference,holdoutPairs:holdout};
}
module.exports={run,fixture};
if(require.main===module)console.log(JSON.stringify(run(),null,2));
