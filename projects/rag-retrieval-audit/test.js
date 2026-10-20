"use strict";
const {Index,tokenize,retrievalMetrics}=require("./core");
function check(x,m){if(!x)throw Error(m);}
function runTests(){
 let count=0;const test=f=>{f();count++;},docs=[{id:"a",text:"key value cache cache"},{id:"b",text:"sparse expert routing"}],index=new Index(docs);
 for(const method of ["bm25","tfidf","overlap"])test(()=>check(index.search("cache",method)[0].id==="a","Relevant document"));
 test(()=>check(index.search("unknown")[0].score===0&&index.search("")[0].score===0,"Zero overlap"));
 test(()=>{let failed=false;try{new Index([{id:"x",text:"a"},{id:"x",text:"b"}]);}catch(e){failed=true;}check(failed,"Duplicate ids");});
 test(()=>check(tokenize("KV-Cache, CACHE").join(" ")==="kv cache cache","Tokenizer"));
 test(()=>{let m=retrievalMetrics({q:[{id:"b"},{id:"a"}]},[{id:"q",relevant:["a"]}]);check(m.recallAt1===0&&m.recallAt3===1&&m.mrr===.5,"Ranking metrics");});
 test(()=>{const a=new Index([{id:"a",text:"word"},{id:"b",text:"word"}]);check(a.search("word")[0].id==="a","Stable ties");});
 return {passed:count};
}
module.exports={runTests};
if(require.main===module)console.log(JSON.stringify(runTests()));
