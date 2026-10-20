"use strict";
const escape=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
function bars(title,labels,values,{maximum=1,unit="",log=false}={}){
 const W=820,H=410,L=95,T=65,PW=690,PH=245;
 const transformed=values.map(v=>log?Math.log2(v):v),max=log?Math.ceil(Math.max(...transformed)/4)*4:maximum;
 let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"><title>${escape(title)}</title><desc>${escape(labels.map((x,i)=>x+": "+values[i]+unit).join("; "))}</desc><rect width="100%" height="100%" fill="#fff"/><g font-family="Arial, sans-serif" fill="#18324a"><text x="35" y="30" font-size="19" font-weight="bold">${escape(title)}</text>`;
 for(let i=0;i<=4;i++){let y=T+PH-i*PH/4,val=max*i/4;s+=`<line x1="${L}" x2="${L+PW}" y1="${y}" y2="${y}" stroke="#dce3e9"/><text x="${L-12}" y="${y+5}" text-anchor="end" font-size="12">${log?"2^"+val:escape(Number(val.toPrecision(3))+unit)}</text>`;}
 const step=PW/labels.length,bw=step*.58;
 labels.forEach((label,i)=>{const h=transformed[i]/max*PH,x=L+i*step+(step-bw)/2,y=T+PH-h;const shown=log?values[i].toLocaleString("en-US"):Number(values[i].toPrecision(4))+unit;
 s+=`<rect x="${x}" y="${y}" width="${bw}" height="${Math.max(h,0)}" fill="#30658a"/><text x="${x+bw/2}" y="${Math.max(y-9,T-10)}" text-anchor="middle" font-size="12">${escape(shown)}</text><text x="${x+bw/2}" y="${T+PH+24}" text-anchor="middle" font-size="12">${escape(label)}</text>`;});
 s+=`<text x="${L}" y="390" font-size="12" fill="#526579">${log?"Logarithmic axis; values are actual bytes.":"Fixed experiment; see metrics and limitations in README."}</text></g></svg>`;
 if(s.includes("NaN")||s.includes("undefined")||!s.endsWith("</svg>"))throw Error("Invalid chart");
 return s;
}
module.exports={bars};
