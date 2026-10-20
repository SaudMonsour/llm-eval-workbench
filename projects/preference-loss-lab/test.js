"use strict";
const {dpoLossGrad,evaluate,probabilities}=require("./core"),{fixture}=require("./study");
function check(x,m){if(!x)throw Error(m);}
function runTests(){
 const ref=[[0,0],[0,0],[0,0]],p=[{x:[.3,-.7],chosen:0,rejected:2}],w=[[.2,-.1],[.3,.1],[-.4,.2]];
 let count=0;const test=f=>{f();count++;};
 test(()=>check(Math.abs(dpoLossGrad(ref,ref,p).loss-Math.log(2))<1e-12,"Reference loss"));
 test(()=>{const g=dpoLossGrad(w,ref,p,.7).grad;for(let a=0;a<3;a++)for(let j=0;j<2;j++){let orig=w[a][j];w[a][j]=orig+1e-6;let hi=dpoLossGrad(w,ref,p,.7).loss;w[a][j]=orig-1e-6;let lo=dpoLossGrad(w,ref,p,.7).loss;w[a][j]=orig;check(Math.abs(g[a][j]-(hi-lo)/2e-6)<1e-8,"Gradient mismatch");}});
 test(()=>{const before=dpoLossGrad(w,ref,p);const updated=w.map((row,a)=>row.map((v,j)=>v-.1*before.grad[a][j]));check(dpoLossGrad(updated,ref,p).loss<before.loss,"No improvement");});
 test(()=>{let shifted=w.map(row=>row.map((v,j)=>v+[2,-3][j]));check(Math.abs(dpoLossGrad(w,ref,p).loss-dpoLossGrad(shifted,ref,p).loss)<1e-12,"Logit shift");});
 test(()=>{let prob=probabilities(w,[100,-100]);check(prob.every(Number.isFinite)&&Math.abs(prob.reduce((a,b)=>a+b,0)-1)<1e-12,"Softmax");});
 test(()=>check(evaluate(ref,ref,p).meanKL===0,"Reference KL"));
 test(()=>{let failed=false;try{dpoLossGrad(w,ref,p,0);}catch(e){failed=true;}check(failed,"Zero beta accepted");});
 test(()=>check(JSON.stringify(fixture(5,41))===JSON.stringify(fixture(5,41))&&JSON.stringify(fixture(5,41))!==JSON.stringify(fixture(5,43)),"Fixture determinism"));
 return {passed:count};
}
module.exports={runTests};
if(require.main===module)console.log(JSON.stringify(runTests()));
