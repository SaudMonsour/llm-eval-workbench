"use strict";
/** Scalar CPU reference for stable, causal attention. */
function validate(q,k,v,n,d,dv,block) {
  if (![n,d,dv,block].every(x=>Number.isInteger(x)&&x>0) ||
      q.length!==n*d || k.length!==n*d || v.length!==n*dv ||
      ![q,k,v].every(a=>Array.from(a).every(Number.isFinite))) {
    throw new Error("Finite arrays and matching positive dimensions required");
  }
}
function dense(q,k,v,n,d,dv,causal=true) {
  validate(q,k,v,n,d,dv,1);
  const scores=new Float64Array(n*n), output=new Float64Array(n*dv);
  for(let i=0;i<n;i++) {
    let max=-Infinity;
    for(let j=0;j<n;j++) {
      let dot=0;
      for(let a=0;a<d;a++) dot+=q[i*d+a]*k[j*d+a];
      const s=causal&&j>i ? -Infinity : dot/Math.sqrt(d);
      scores[i*n+j]=s; max=Math.max(max,s);
    }
    let denominator=0;
    for(let j=0;j<n;j++) denominator+=(scores[i*n+j]=Math.exp(scores[i*n+j]-max));
    for(let j=0;j<n;j++) for(let a=0;a<dv;a++)
      output[i*dv+a]+=scores[i*n+j]/denominator*v[j*dv+a];
  }
  return {output,scratchBytes:scores.byteLength};
}
function online(q,k,v,n,d,dv,block=16,causal=true) {
  validate(q,k,v,n,d,dv,block);
  const scores=new Float64Array(block), numerator=new Float64Array(dv), output=new Float64Array(n*dv);
  for(let i=0;i<n;i++) {
    let max=-Infinity, denominator=0; numerator.fill(0);
    const visible=causal ? i+1 : n;
    for(let start=0;start<visible;start+=block) {
      const count=Math.min(block,visible-start); let blockMax=-Infinity;
      for(let b=0;b<count;b++) {
        let dot=0,j=start+b;
        for(let a=0;a<d;a++) dot+=q[i*d+a]*k[j*d+a];
        scores[b]=dot/Math.sqrt(d); blockMax=Math.max(blockMax,scores[b]);
      }
      const nextMax=Math.max(max,blockMax), correction=Math.exp(max-nextMax);
      denominator*=correction;
      for(let a=0;a<dv;a++) numerator[a]*=correction;
      for(let b=0;b<count;b++) {
        const weight=Math.exp(scores[b]-nextMax); denominator+=weight;
        for(let a=0;a<dv;a++) numerator[a]+=weight*v[(start+b)*dv+a];
      }
      max=nextMax;
    }
    for(let a=0;a<dv;a++) output[i*dv+a]=numerator[a]/denominator;
  }
  return {output,scratchBytes:scores.byteLength+numerator.byteLength};
}
module.exports={dense,online};
