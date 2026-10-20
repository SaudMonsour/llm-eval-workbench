"use strict";
const {dense,online}=require("./core"),{fixture}=require("./study");
function check(x,m){if(!x)throw Error(m);}
function near(a,b){check(a.length===b.length&&a.every((x,i)=>Math.abs(x-b[i])<1e-10),"Outputs differ");}
function runTests(){
 let count=0;const test=f=>{f();count++;};
 for(const block of [1,3,16]) test(()=>{let f=fixture(11);near(dense(f.q,f.k,f.v,f.n,f.d,f.dv).output,online(f.q,f.k,f.v,f.n,f.d,f.dv,block).output);});
 test(()=>{let f=fixture(9,4,3,100);near(dense(f.q,f.k,f.v,9,4,3).output,online(f.q,f.k,f.v,9,4,3,3).output);});
 test(()=>{let f=fixture(7,4,2);let a=online(f.q,f.k,f.v,7,4,2).output;f.v.fill(100,8);let b=online(f.q,f.k,f.v,7,4,2).output;near(a.slice(0,8),b.slice(0,8));});
 test(()=>{let f=fixture(7,4,2);near(dense(f.q,f.k,f.v,7,4,2,false).output,online(f.q,f.k,f.v,7,4,2,4,false).output);});
 test(()=>{let f=fixture(1,4,2);near(online(f.q,f.k,f.v,1,4,2).output,f.v);});
 test(()=>{let f=fixture(3);f.q[0]=NaN;let failed=false;try{online(f.q,f.k,f.v,3,32,16);}catch(e){failed=true;}check(failed,"Nonfinite accepted");});
 return {passed:count};
}
module.exports={runTests};
if(require.main===module)console.log(JSON.stringify(runTests()));
