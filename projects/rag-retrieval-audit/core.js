"use strict";
const STOP=new Set("a an the and or of to in is it for with how which what does can from this that".split(" "));
function tokenize(text){if(typeof text!=="string")throw Error("Text required");return text.normalize("NFKC").toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(x=>x&&!STOP.has(x));}
function counts(tokens){const m=new Map();for(const t of tokens)m.set(t,(m.get(t)||0)+1);return m;}
class Index{
 constructor(docs){
  if(!Array.isArray(docs)||!docs.length||new Set(docs.map(d=>d.id)).size!==docs.length||!docs.every(d=>typeof d.id==="string"&&d.id&&typeof d.text==="string"))throw Error("Unique ids and text required");
  this.docs=docs.map(d=>({...d,tf:counts(tokenize(d.text)),length:tokenize(d.text).length}));
  this.df=new Map();for(const d of this.docs)for(const t of d.tf.keys())this.df.set(t,(this.df.get(t)||0)+1);
  this.averageLength=this.docs.reduce((s,d)=>s+d.length,0)/docs.length||1;
 }
 idf(t){return Math.log((this.docs.length+1)/((this.df.get(t)||0)+1))+1;}
 search(query,method="bm25",k=this.docs.length){
  if(!["bm25","tfidf","overlap"].includes(method)||!Number.isInteger(k)||k<1)throw Error("Invalid search configuration");
  const terms=counts(tokenize(query)),N=this.docs.length;
  let qnorm=0;for(const [t,c]of terms)qnorm+=((1+Math.log(c))*this.idf(t))**2;qnorm=Math.sqrt(qnorm);
  return this.docs.map(d=>{
   let score=0,norm=0;
   if(method==="tfidf")for(const [t,c]of d.tf)norm+=((1+Math.log(c))*this.idf(t))**2;
   for(const [t,qtf]of terms){
    const tf=d.tf.get(t)||0;if(!tf)continue;
    if(method==="bm25"){const df=this.df.get(t);score+=Math.log(1+(N-df+.5)/(df+.5))*tf*2.2/(tf+1.2*(.25+.75*d.length/this.averageLength));}
    else if(method==="overlap")score++;
    else score+=(1+Math.log(qtf))*this.idf(t)*(1+Math.log(tf))*this.idf(t);
   }
   if(method==="tfidf")score=norm&&qnorm?score/Math.sqrt(norm)/qnorm:0;
   return {id:d.id,score};
  }).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id)).slice(0,k);
 }
}
function retrievalMetrics(rankings,queries){
 const answerable=queries.filter(q=>q.relevant.length);let hit1=0,hit3=0,mrr=0;
 for(const q of answerable){const ranked=rankings[q.id],rank=ranked.findIndex(d=>q.relevant.includes(d.id));hit1+=rank===0?1:0;hit3+=rank>=0&&rank<3?1:0;mrr+=rank>=0?1/(rank+1):0;}
 return {answerableQueries:answerable.length,recallAt1:hit1/answerable.length,recallAt3:hit3/answerable.length,mrr:mrr/answerable.length};
}
module.exports={Index,tokenize,retrievalMetrics};
