"use strict";
const {dense,online}=require("./core");
function rng(seed){return()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};}
function fixture(n,d=32,dv=16,scale=1) {
 const random=rng(42), make=(len,s)=>Float64Array.from({length:len},()=>s*(random()*2-1));
 return {q:make(n*d,scale),k:make(n*d,scale),v:make(n*dv,1),n,d,dv};
}
function run(){
 const rows=[];
 for(const n of [64,128,256]) for(const scale of [1,40]){
  const {q,k,v,d,dv}=fixture(n,32,16,scale),ref=dense(q,k,v,n,d,dv);
  for(const block of [1,8,32]){
   const result=online(q,k,v,n,d,dv,block);
   const error=Math.max(...Array.from(ref.output,(x,i)=>Math.abs(x-result.output[i])));
   rows.push({tokens:n,scale,block,maxAbsError:error,denseScratchBytes:ref.scratchBytes,onlineScratchBytes:result.scratchBytes,scratchReduction:ref.scratchBytes/result.scratchBytes});
  }
 }
 return {seed:42,fixture:"Deterministic synthetic Q/K/V, float64, causal CPU reference",metricScope:"Allocated numeric scratch buffers only; inputs, outputs, JS overhead and GPU memory excluded",results:rows};
}
module.exports={run,fixture};
if(require.main===module) console.log(JSON.stringify(run(),null,2));
