/* WaveCut audio engine. Independently implemented DSP; not WaveMix source. */
(function(root){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const mean=a=>a.reduce((s,x)=>s+x,0)/Math.max(1,a.length);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
function norm(a){const n=Math.sqrt(dot(a,a))||1;return a.map(x=>x/n);}
function similarity(a,b){return clamp(dot(a,b),0,1);}
function fft(real,imag){const n=real.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[real[i],real[j]]=[real[j],real[i]];[imag[i],imag[j]]=[imag[j],imag[i]];}}for(let len=2;len<=n;len<<=1){const angle=-2*Math.PI/len,wr0=Math.cos(angle),wi0=Math.sin(angle);for(let i=0;i<n;i+=len){let wr=1,wi=0;for(let j=0;j<len/2;j++){const x=i+j,y=x+len/2,tr=real[y]*wr-imag[y]*wi,ti=real[y]*wi+imag[y]*wr;real[y]=real[x]-tr;imag[y]=imag[x]-ti;real[x]+=tr;imag[x]+=ti;const next=wr*wr0-wi*wi0;wi=wr*wi0+wi*wr0;wr=next;}}}}
function analyze(samples,rate,options={},progress=()=>{}){
 if(!samples.length)throw new Error('音频为空');
 const n=2048,hop=256,frames=Math.max(1,Math.ceil((samples.length-n)/hop)+1),dt=hop/rate;
 const rms=new Float32Array(frames),flux=new Float32Array(frames),chroma=[],timbre=[];
 const re=new Float64Array(n),im=new Float64Array(n),previous=new Float64Array(n/2),hann=Float64Array.from({length:n},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(n-1)));
 let maxRms=0;
 for(let f=0;f<frames;f++){
  let sq=0;for(let j=0;j<n;j++){const s=samples[f*hop+j]||0;sq+=s*s;re[j]=s*hann[j];im[j]=0;}rms[f]=Math.sqrt(sq/n);maxRms=Math.max(maxRms,rms[f]);fft(re,im);
  const pc=Array(12).fill(0),bands=Array(18).fill(0);let onset=0;
  for(let k=1;k<n/2;k++){const hz=k*rate/n,mag=Math.sqrt(re[k]*re[k]+im[k]*im[k]),compressed=Math.log1p(mag);onset+=Math.max(0,compressed-previous[k]);previous[k]=compressed;if(hz>=65&&hz<=2200){const midi=69+12*Math.log2(hz/440);const nearest=Math.round(midi);pc[((nearest%12)+12)%12]+=mag*mag*Math.max(.1,1-Math.abs(midi-nearest));}const band=clamp(Math.floor(Math.log2(Math.max(40,hz)/40)*2.4),0,17);bands[band]+=mag*mag;}
  flux[f]=onset;chroma.push(norm(pc.map(Math.sqrt)));timbre.push(norm(bands.map(x=>Math.log1p(x))));
  if(f%200===0)progress({stage:'扫描全曲音色与节奏',progress:Math.round(f/frames*60)});
 }
 const duration=samples.length/rate,threshold=Math.max(.00015,maxRms*.006);let first=0,last=frames-1;while(first<last&&rms[first]<threshold)first++;while(last>first&&rms[last]<threshold)last--;
 const activeStart=Math.max(0,first*dt-.05),activeEnd=Math.min(duration,(last*hop+n)/rate+.12);
 if(maxRms<.0001)throw new Error('这段音频几乎全是静音，无法分析音乐结构。');
 const novelty=Float32Array.from(flux,(v,i)=>Math.max(0,v-mean(Array.from(flux.subarray(Math.max(0,i-8),Math.min(frames,i+9))))));
 let bestTempo=120,bestScore=-1,secondScore=-1;
 for(let bpm=65;bpm<=180;bpm+=.25){const lag=60/bpm/dt;let sum=0,den=0;for(let i=Math.ceil(lag);i<frames;i++){const x=i-lag,lo=Math.floor(x),v=novelty[lo]*(1-(x-lo))+(novelty[lo+1]||0)*(x-lo);sum+=novelty[i]*v;den+=novelty[i]*novelty[i];}const score=sum/Math.max(1e-9,den)*(.92+.08*Math.exp(-(((bpm-115)/45)**2)));if(score>bestScore){secondScore=bestScore;bestScore=score;bestTempo=bpm;}}
 if(options.bpm)bestTempo=clamp(options.bpm,40,240);
 const period=60/bestTempo;let phase=0,phaseScore=-1;
 for(let offset=0;offset<period;offset+=dt/2){let sum=0;for(let t=offset;t<duration;t+=period){const f=Math.round(t/dt);sum+=Math.max(...Array.from(novelty.subarray(Math.max(0,f-1),Math.min(frames,f+2))),0);}if(sum>phaseScore){phaseScore=sum;phase=offset;}}
 const beats=[];for(let t=phase;t<activeEnd;t+=period)if(t>=activeStart-.03)beats.push(Number(t.toFixed(5)));
 let barPhase=0,barScore=-1;const meter=options.meter===3?3:4;
 for(let p=0;p<meter;p++){let score=0;for(let i=p;i<beats.length;i+=meter){const f=clamp(Math.round(beats[i]/dt),0,frames-1);score+=novelty[f]+flux[f]*.15;}if(score>barScore){barScore=score;barPhase=p;}}
 // The global tempo grid can drift by a tenth of a second over a long song.
 // Pull each candidate downbeat toward a nearby local attack before cutting.
 const bars=beats.filter((_,i)=>i%meter===barPhase).map(t=>{
  const radius=Math.min(.18,period*.34),lo=clamp(Math.floor((t-radius)/dt),0,frames-1),hi=clamp(Math.ceil((t+radius)/dt),lo,frames-1);
  let localPeak=0;for(let f=lo;f<=hi;f++)localPeak=Math.max(localPeak,novelty[f]);
  if(localPeak<=0)return t;
  let best=clamp(Math.round(t/dt),lo,hi),bestScore=-Infinity;
  for(let f=lo;f<=hi;f++){const score=novelty[f]/localPeak-Math.abs(f*dt-t)/radius*.2;if(score>bestScore){bestScore=score;best=f;}}
  return Number(clamp(best*dt-.012,activeStart,activeEnd).toFixed(5));
 });
 const featureAt=(t,span=.4)=>{const a=clamp(Math.floor(t/dt),0,frames-1),b=clamp(Math.ceil((t+span)/dt),a+1,frames);const pc=Array(12).fill(0),tb=Array(18).fill(0);let energy=0;for(let f=a;f<b;f++){for(let j=0;j<12;j++)pc[j]+=chroma[f][j];for(let j=0;j<18;j++)tb[j]+=timbre[f][j];energy+=rms[f];}return {chroma:norm(pc),timbre:norm(tb),energy:energy/(b-a)};};
 const barFeatures=bars.map(t=>featureAt(t,period*meter));
 const changes=bars.map((t,i)=>{if(!i)return 0;const a=barFeatures[i-1],b=barFeatures[i];return 1-similarity(a.chroma,b.chroma)+(1-similarity(a.timbre,b.timbre))*.6+Math.min(1,Math.abs(Math.log((a.energy+.0001)/(b.energy+.0001))))*.3;});
 const sortedChanges=[...changes].sort((a,b)=>a-b),cutoff=Math.max(.12,sortedChanges[Math.floor(sortedChanges.length*.72)]||.12);
 const bounds=[activeStart];for(let i=1;i<bars.length;i++){const t=bars[i],gap=t-bounds[bounds.length-1];if(gap>=period*meter*2&&activeEnd-t>=period*meter&&(changes[i]>=cutoff||gap>=period*meter*8))bounds.push(t);}bounds.push(activeEnd);
 const sectionFeatures=bounds.slice(0,-1).map((t,i)=>featureAt(t,bounds[i+1]-t));const energies=sectionFeatures.map(x=>x.energy),energyCut=[...energies].sort((a,b)=>a-b)[Math.floor(energies.length*.6)]||0;
 const sections=sectionFeatures.map((f,i)=>{let repeats=0;sectionFeatures.forEach((g,j)=>{if(Math.abs(i-j)>1&&similarity(f.chroma,g.chroma)>.9&&similarity(f.timbre,g.timbre)>.92)repeats++;});let label='段落';if(i===0)label='前奏候选';else if(i===sectionFeatures.length-1)label='尾奏候选';else if(repeats&&f.energy>=energyCut)label='副歌候选';else if(f.energy<energyCut*.6)label='间奏候选';else label='主歌候选';return {start:bounds[i],end:bounds[i+1],label,energy:f.energy,repeats,estimated:true};});
 const global=norm(Array.from({length:12},(_,j)=>chroma.reduce((s,c)=>s+c[j],0)));
 const major=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],minor=[6.33,2.68,3.52,5.38,2.6,3.53,2.54,4.75,3.98,2.69,3.34,3.17];let key='未知',keyScore=-1;const keys=['C','C♯','D','E♭','E','F','F♯','G','A♭','A','B♭','B'];for(let tonic=0;tonic<12;tonic++)for(const [profile,name]of[[major,'大调'],[minor,'小调']]){const score=dot(global,norm(profile.map((_,j)=>profile[(j-tonic+12)%12])));if(score>keyScore){keyScore=score;key=keys[tonic]+' '+name;}}
 progress({stage:'全曲分析完成',progress:100});
 return {version:3,method:'spectral-dsp',duration,activeStart,activeEnd,bpm:Math.round(bestTempo*10)/10,beatConfidence:clamp(bestScore,0,1),meter,meterEstimated:!options.meter,key,keyEstimated:true,beats,bars,sections,frames:{dt,rms:Array.from(rms),onsets:Array.from(novelty),chroma,timbre},maxRms};
}
function frameFeature(analysis,t){const i=clamp(Math.round(t/analysis.frames.dt),0,analysis.frames.rms.length-1);return {chroma:analysis.frames.chroma[i],timbre:analysis.frames.timbre[i],energy:analysis.frames.rms[i]};}
function seamCost(a,from,to,lyrics=[]){
 // Compare a short musical context on both sides of a downbeat. A single FFT
 // frame can match by accident while the following phrase sounds unrelated.
 const beat=60/a.bpm, offsets=[-.75,-.35,.1,.35,.75,1.25].map(x=>x*beat);
 let harmonic=0,texture=0,level=0;
 for(let i=0;i<offsets.length;i++){
  const left=frameFeature(a,from+offsets[i]),right=frameFeature(a,to+offsets[i]);
  const weight=i===2?1.4:1;
  harmonic+=(1-similarity(left.chroma,right.chroma))*weight;
  texture+=(1-similarity(left.timbre,right.timbre))*weight;
  level+=Math.min(2,Math.abs(Math.log((left.energy+.0001)/(right.energy+.0001))))*weight;
 }
 const total=6.4;let rhythm=0;
 if(a.frames.onsets){
  const envelope=t=>Array.from({length:12},(_,i)=>{const f=clamp(Math.round((t+i*beat/4)/a.frames.dt),0,a.frames.onsets.length-1);return Math.max(...a.frames.onsets.slice(Math.max(0,f-2),Math.min(a.frames.onsets.length,f+3)));});
  rhythm=1-similarity(norm(envelope(from)),norm(envelope(to)));
 }
 let vocal=0;
 for(const line of lyrics){if(from>line.start+.08&&from<line.end-.08)vocal+=2;if(to>line.start+.08&&to<line.end-.08)vocal+=2;}
 return harmonic/total*2.1+texture/total*.95+level/total*.32+rhythm*.7+vocal;
}
function buildPieces(start,end,skips){const pieces=[];let at=start;for(const cut of [...skips].sort((a,b)=>a.from-b.from)){if(cut.from>at)pieces.push({start:at,end:cut.from,track:0});at=cut.to;}if(at<end)pieces.push({start:at,end,track:0});return pieces;}
function planDuration(pieces){return pieces.reduce((s,p)=>s+(p.end-p.start)/(p.rate||1),0);}
function makePlans(a,target,options={}){
 if(!Number.isFinite(target)||target<5||target>1200)throw new Error('目标时长请输入 5–1200 秒。');
 const start=a.activeStart,end=a.activeEnd,duration=end-start,bar=60/a.bpm*a.meter,lyrics=options.lyrics||[];
 const strict=options.exact===true,candidates=[];
 function add(pieces,name,cost,kind){const natural=planDuration(pieces),ratio=target/natural;if(strict&&(ratio<.94||ratio>1.06))return;if(!strict&&Math.abs(natural-target)>.85)return;const seams=pieces.length-1;const retainedChorus=a.sections.filter(s=>/副歌/.test(s.label)).filter(s=>pieces.some(p=>p.end>=s.end&&p.start<=s.start)).length;const headShare=(pieces[0].end-pieces[0].start)/Math.max(1,natural);const boundaries=a.sections.flatMap(s=>[s.start,s.end]);const distance=t=>Math.min(...boundaries.map(b=>Math.abs(b-t)));const structurePenalty=pieces.slice(1).reduce((sum,p,i)=>sum+Math.min(1,distance(pieces[i].end)/Math.max(.25,bar*.25))*.7+Math.min(1,distance(p.start)/Math.max(.25,bar*.25))*.7,0);const score=cost+Math.abs(natural-target)/Math.max(1,target)*8+seams*.1-retainedChorus*.12+Math.abs(headShare-.4)*.45+structurePenalty;const signature=pieces.map(p=>[p.start.toFixed(2),p.end.toFixed(2)].join(':')).join('|');if(candidates.some(p=>p.signature===signature))return;candidates.push({id:'p'+candidates.length,name,kind,pieces,naturalDuration:natural,duration:strict?target:natural,stretch:strict?ratio:1,seams,cost:score,signature,retainedChorus,openingPreserved:Math.abs(pieces[0].start-start)<.05,endingPreserved:true,lyricProtected:lyrics.length>0});}
 if(Math.abs(duration-target)<.03){add([{start,end,track:0}],'保留完整原曲',0,'original');return candidates;}
 const minHead=Math.min(bar, target*.22),minTail=Math.min(bar,target*.25),points=[...new Set([...a.bars,...a.sections.slice(1).map(s=>a.beats.reduce((best,b)=>Math.abs(b-s.start)<Math.abs(best-s.start)?b:best,a.beats[0]||s.start))])].filter(t=>t>start+minHead&&t<end-minTail).sort((x,y)=>x-y);
 if(target<duration){
  if(target<=duration*.55){const tailStarts=[...new Set([...a.bars,...a.sections.map(s=>s.start)])].filter(t=>t>start+bar*.75&&Math.abs(end-t-target)<=.85).sort((x,y)=>Math.abs(end-x-target)-Math.abs(end-y-target));for(const at of tailStarts.slice(0,2))add([{start:at,end,track:0}],'连续尾段 · 无拼接',.8,'continuous-tail');}
  const remove=duration-target,tolerance=Math.max(bar*1.1,target*.045),cuts=[];for(let i=0;i<lyrics.length;i++)for(let j=i+1;j<lyrics.length;j++){const clean=s=>s.toLowerCase().replace(/[\s\p{P}]/gu,'');if(clean(lyrics[i].text).length<3||clean(lyrics[i].text)!==clean(lyrics[j].text))continue;const from=lyrics[i].start,to=lyrics[j].start;if(from>start+minHead&&to<end-minTail&&Math.abs(to-from-remove)<=tolerance)add(buildPieces(start,end,[{from,to}]),'重复歌词跳接 · 同句接同句',seamCost(a,from,to,[])-.15,'lyric-repeat');}
  for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){const from=points[i],to=points[j],removed=to-from;if(removed>remove+tolerance)break;const cost=seamCost(a,from,to,lyrics);cuts.push({from,to,removed,cost});if(Math.abs(removed-remove)<=tolerance)add(buildPieces(start,end,[{from,to}]),'一处衔接 · 保留起收',cost,'shorten');}
  // Two separated skips offer an alternative to deleting one long passage.
  const partial=cuts.filter(c=>c.removed<remove-bar&&c.removed>bar*.9).sort((x,y)=>x.cost-y.cost).slice(0,180);
  for(const x of partial){let best=null;for(const y of cuts){if(y.from<x.to+bar)continue;const error=Math.abs(x.removed+y.removed-remove);if(error>tolerance)continue;const cost=x.cost+y.cost+error/Math.max(1,bar)*.2;if(!best||cost<best.cost)best={y,cost};}if(best)add(buildPieces(start,end,[x,best.y]),'两处衔接 · 分段压缩',best.cost,'shorten');}
  if(!candidates.length){const keepTail=Math.min(target*.4,Math.max(2,bar));const head=target-keepTail;add([{start,end:start+head,track:0},{start:end-keepTail,end,track:0}],'短时长备选 · 接缝需核对',4,'shorten');}
 }else{
  const extra=target-duration;
  for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){const from=points[i],to=points[j],loop=to-from;if(loop<bar*2||loop>duration*.7)continue;const count=clamp(Math.round(extra/loop),1,32),natural=duration+loop*count;if(Math.abs(natural-target)>Math.max(bar*1.1,target*.06))continue;const pieces=[{start,end:to,track:0}];for(let k=0;k<count;k++)pieces.push({start:from,end:to,track:0});pieces.push({start:to,end,track:0});add(pieces,'乐段复现 · 延长后自然收尾',seamCost(a,to,from,lyrics)*count*.5,'extend');}
  if(!candidates.length){const from=start+Math.min(bar,duration*.15),to=end-Math.min(bar,duration*.15),loop=to-from,count=Math.max(1,Math.round(extra/loop));const pieces=[{start,end:to,track:0}];for(let k=0;k<count;k++)pieces.push({start:from,end:to,track:0});pieces.push({start:to,end,track:0});add(pieces,'循环乐段备选',3,'extend');}
 }
 candidates.sort((x,y)=>x.cost-y.cost);const selected=[];
 for(const p of candidates){if(selected.length>=(options.limit||4))break;if(selected.some(q=>q.pieces.length===p.pieces.length&&Math.abs(q.pieces[0].end-p.pieces[0].end)<bar*.7))continue;selected.push(p);}
 for(let i=0;i<selected.length;i++){selected[i].id='plan-'+i;selected[i].label=i===0?'推荐试听':`备选 ${i}`;selected[i].risk=selected[i].cost>2.2?'接缝风险较高':selected[i].cost>1?'需要试听确认':'接缝特征较接近';}
 if(!selected.length)throw new Error('当前目标无法在保留起收的同时得到合理方案，请调整几秒后重试。');
 return selected;
}
function mixPlans(a,b,target,options={}){
 const ratio=b.bpm/a.bpm,tempoRate=clamp(ratio,.75,1.33),bRate=1/tempoRate,candidates=[];
 const start=a.activeStart,end=b.activeEnd,fade=clamp(options.crossfade||2,.05,8);
 for(const cutA of a.bars){if(cutA<start+4)continue;const desiredB=end-(target-(cutA-start))*bRate;for(const cutB of b.bars){if(Math.abs(cutB-desiredB)>60/b.bpm*4||cutB>end-4)continue;const fa=frameFeature(a,cutA),fb=frameFeature(b,cutB);const cost=1-similarity(fa.chroma,fb.chroma)+.5*(1-similarity(fa.timbre,fb.timbre))+.15*Math.abs(Math.log((fa.energy+.001)/(fb.energy+.001)));const pieces=[{start,end:cutA,track:0},{start:cutB,end,track:1,rate:bRate}],natural=planDuration(pieces),stretch=target/natural;if(options.exact!==false&&(stretch<.9||stretch>1.1))continue;candidates.push({id:'mix-'+candidates.length,name:'A → B · 节拍与响度衔接',kind:'mix',pieces,naturalDuration:natural,duration:options.exact===false?natural:target,stretch:options.exact===false?1:stretch,seams:1,cost:cost+Math.abs(target-natural)/target*10,endingPreserved:true,crossfade:fade,bRate,risk:'两首歌曲仍需试听核对调性和过渡'});}}
 candidates.sort((x,y)=>x.cost-y.cost);return candidates.filter((p,i,all)=>i===0||Math.abs(p.pieces[0].end-all[0].pieces[0].end)>60/a.bpm*2).slice(0,4).map((p,i)=>({...p,label:i?'备选 '+i:'推荐试听'}));
}
function boundaryCost(a,from,to){
 const avg=times=>{const chroma=Array(12).fill(0),timbre=Array(18).fill(0);let energy=0;for(const t of times){const f=frameFeature(a,t);for(let i=0;i<12;i++)chroma[i]+=f.chroma[i];for(let i=0;i<18;i++)timbre[i]+=f.timbre[i];energy+=f.energy;}return {chroma:norm(chroma),timbre:norm(timbre),energy:energy/times.length};};
 const before=avg([-.42,-.28,-.14].map(x=>from+x)),after=avg([.12,.26,.4].map(x=>to+x));
 return (1-similarity(before.chroma,after.chroma))*1.6+(1-similarity(before.timbre,after.timbre))*.8+Math.min(2,Math.abs(Math.log((before.energy+.0001)/(after.energy+.0001))))*.3;
}
function grooveSimilarity(a,from,to){
 const onset=a.frames.onsets;if(!onset)return 0;
 const beat=60/a.bpm,dt=a.frames.dt;
 const pattern=t=>Array.from({length:16},(_,i)=>{const f=clamp(Math.round((t+(i-4)*beat/4)/dt),0,onset.length-1);let peak=0;for(let j=Math.max(0,f-2);j<=Math.min(onset.length-1,f+2);j++)peak=Math.max(peak,onset[j]);return peak;});
 return similarity(norm(pattern(from)),norm(pattern(to)));
}
function fadeRetention(source,rate,from,to,fade){
 const n=Math.max(16,Math.round(fade*rate)),a=Math.round(from*rate)-n,b=Math.round(to*rate)-n;
 if(a<0||b<0||a+n>=source.length||b+n>=source.length)return 1;
 let mixed=0,expected=0;
 for(let i=0;i<n;i+=4){const t=(i+1)/(n+1),x=source[a+i],y=source[b+i],v=x*(1-t)+y*t;mixed+=v*v;expected+=(x*x*(1-t)*(1-t)+y*y*t*t);}
 return expected<1e-7?1:Math.sqrt(mixed/expected);
}
function assessPlans(a,plans,source,rate,options={}){
 const natural=a.bars.filter(t=>t>a.activeStart+.6&&t<a.activeEnd-.6).map(t=>boundaryCost(a,t,t)).sort((x,y)=>x-y);
 const reference=natural[Math.floor(natural.length*.85)]||.2,limit=Math.max(.34,reference*1.45+.08),safeLimit=Math.min(limit,Math.max(.45,reference*1.05)),fade=options.crossfade>0?clamp(options.crossfade,.005,.5):.025;
 const level=t=>{const first=Math.max(0,Math.round(t*rate)),last=Math.min(source.length,first+Math.round(rate*.5));let sum=0,count=0;for(let i=first;i<last;i+=8){sum+=source[i]*source[i];count++;}return Math.sqrt(sum/Math.max(1,count));};
 return plans.map(plan=>{
  const joins=[];for(let i=1;i<plan.pieces.length;i++){const from=plan.pieces[i-1].end,to=plan.pieces[i].start,change=boundaryCost(a,from,to),groove=grooveSimilarity(a,from,to),fadeLevel=fadeRetention(source,rate,from,to,fade),left=level(from-.6),right=level(to+.1),levelDb=20*Math.log10((right+1e-5)/(left+1e-5));joins.push({from,to,change,groove,fadeLevel,levelDb,passed:change<=safeLimit&&groove>=.9&&fadeLevel>=.55&&Math.abs(levelDb)<=3});}
  const passed=joins.every(j=>j.passed),penalty=joins.reduce((sum,j)=>sum+j.change/limit*.4+(1-j.groove)*.4+Math.max(0,1-j.fadeLevel)*.7,0);
  return {...plan,quality:{passed,joins,limit:safeLimit,score:plan.cost+penalty},risk:plan.seams===0?'无内部拼接 · 请听开头是否自然':passed?'接缝指标通过 · 仍需试听':'接缝检查未通过'};
 }).sort((x,y)=>x.quality.score-y.quality.score);
}
// Waveform-similarity overlap-add. All channels share alignment; pitch is preserved.
function stretch(channels,sampleRate,targetFrames){
 const inputLength=channels[0].length;if(targetFrames===inputLength)return channels.map(c=>c.slice());
 const ratio=targetFrames/inputLength,window=Math.max(128,Math.round(sampleRate*.04)),synthHop=Math.floor(window/2),search=Math.round(sampleRate*.008),weights=Float32Array.from({length:window},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(window-1)));
 const result=channels.map(()=>new Float32Array(targetFrames)),weight=new Float32Array(targetFrames);let previous=0;
 for(let out=0;out<targetFrames;out+=synthHop){const expected=Math.round(out/ratio);let best=clamp(expected,0,Math.max(0,inputLength-window));
  if(out>0&&expected+window+search<inputLength){let bestScore=-Infinity;for(let candidate=Math.max(0,expected-search);candidate<=Math.min(inputLength-window,expected+search);candidate+=4){let cross=0,aa=0,bb=0;for(let k=0;k<synthHop;k+=12){const x=channels[0][Math.min(inputLength-1,previous+synthHop+k)],y=channels[0][candidate+k];cross+=x*y;aa+=x*x;bb+=y*y;}const score=cross/Math.sqrt(aa*bb+1e-12)-Math.abs(candidate-expected)/Math.max(1,search)*.015;if(score>bestScore){bestScore=score;best=candidate;}}}
  for(let k=0;k<window&&out+k<targetFrames;k++){const w=weights[k];weight[out+k]+=w;for(let ch=0;ch<channels.length;ch++)result[ch][out+k]+=(channels[ch][Math.min(inputLength-1,best+k)]||0)*w;}previous=best;
 }
 for(let i=0;i<targetFrames;i++)if(weight[i]>1e-7)for(const c of result)c[i]/=weight[i];
 // Preserve the original opening and tail; the middle absorbs timing correction.
 const edge=Math.min(Math.round(sampleRate*.12),Math.floor(targetFrames/5),Math.floor(inputLength/5));for(let k=0;k<edge;k++)for(let ch=0;ch<channels.length;ch++){const q=k/Math.max(1,edge-1);result[ch][k]=channels[ch][k]*(1-q)+result[ch][k]*q;const j=targetFrames-edge+k;result[ch][j]=result[ch][j]*(1-q)+channels[ch][inputLength-edge+k]*q;}
 return result;
}
function stretchKeepingEdges(channels,rate,targetFrames){
 const inputLength=channels[0].length;
 if(targetFrames===inputLength)return channels.map(c=>c.slice());
 // Keep the opening and, especially, the original ending at their native
 // timing. Correct duration in the middle so the cadence is not smeared.
 const head=Math.min(Math.round(rate*.75),Math.floor(inputLength*.08),Math.floor(targetFrames*.08));
 const tail=Math.min(Math.round(rate*1.5),Math.floor(inputLength*.12),Math.floor(targetFrames*.12));
 const targetMiddle=targetFrames-head-tail;
 if(targetMiddle<rate*.3||inputLength-head-tail<rate*.3)return stretch(channels,rate,targetFrames);
 const middle=stretch(channels.map(c=>c.subarray(head,inputLength-tail)),rate,targetMiddle);
 return channels.map((c,ch)=>{const out=new Float32Array(targetFrames);out.set(c.subarray(0,head),0);out.set(middle[ch],head);out.set(c.subarray(inputLength-tail),head+targetMiddle);return out;});
}
function render(sources,rate,plan,options={}){
 const channels=Math.max(...sources.map(s=>s.length)),pieces=plan.pieces,chunks=[];
 for(const piece of pieces){const source=sources[piece.track||0],a=Math.round(piece.start*rate),b=Math.round(piece.end*rate);let segment=Array.from({length:channels},(_,ch)=>source[Math.min(ch,source.length-1)].slice(a,b));if(piece.rate&&Math.abs(piece.rate-1)>.0001)segment=stretch(segment,rate,Math.round(segment[0].length/piece.rate));chunks.push(segment);}
 const length=chunks.reduce((s,c)=>s+c[0].length,0),out=Array.from({length:channels},()=>new Float32Array(length));
 const rms=(data,start,end)=>{let power=0,count=0;for(const channel of data)for(let i=Math.max(0,start);i<Math.min(channel.length,end);i+=8){power+=channel[i]*channel[i];count++;}return Math.sqrt(power/Math.max(1,count));};
 const beat=60/(options.bpm||plan.bpm||120);let offset=0;const joins=[],transitions=[];
 chunks.forEach((chunk,index)=>{
  if(index){
   const piece=pieces[index],src=sources[piece.track||0],pre=Math.round(piece.start*rate);
   const oldLevel=rms(out,offset-Math.round(rate*.7),offset-Math.round(rate*.1)),newLevel=rms(chunk,Math.round(rate*.1),Math.round(rate*.7));
   const match=oldLevel>.0003&&newLevel>.0003?clamp(oldLevel/newLevel,.8,1.25):1;
   const fade=options.crossfade>0?clamp(options.crossfade,.005,.5):.025;
   const cross=Math.min(Math.round(rate*fade),pre,offset,Math.floor(chunk[0].length/3));
   const release=Math.min(chunk[0].length,Math.round(rate*Math.max(1.2,beat*4)));
   for(const c of chunk)for(let i=0;i<release;i++){const ease=.5+.5*Math.cos(Math.PI*i/release);c[i]*=1+(match-1)*ease;}
   let xy=0,xx=0,yy=0;for(let ch=0;ch<channels;ch++){const incoming=src[Math.min(ch,src.length-1)];for(let k=0;k<cross;k+=8){const x=out[ch][offset-cross+k],y=(incoming[pre-cross+k]||0)*match;xy+=x*y;xx+=x*x;yy+=y*y;}}
   const correlation=clamp(xy/Math.sqrt(xx*yy+1e-12),0,.95);
   for(let k=0;k<cross;k++){const t=(k+1)/(cross+1),oldWeight=Math.cos(Math.PI*t/2),newWeight=Math.sin(Math.PI*t/2),normalizer=Math.sqrt(oldWeight*oldWeight+newWeight*newWeight+2*correlation*oldWeight*newWeight);for(let ch=0;ch<channels;ch++){const prev=out[ch][offset-cross+k],incoming=src[Math.min(ch,src.length-1)][pre-cross+k]||0;out[ch][offset-cross+k]=(prev*oldWeight+incoming*match*newWeight)/normalizer;}}
   joins.push(offset/rate);transitions.push({time:offset/rate,fadeSeconds:cross/rate,levelGainDb:20*Math.log10(match),correlation});
  }
  for(let ch=0;ch<channels;ch++)out[ch].set(chunk[ch],offset);
  offset+=chunk[0].length;
 });
 const frames=Math.round(plan.duration*rate),result=stretchKeepingEdges(out,rate,frames);let peak=0;for(const c of result)for(const x of c)peak=Math.max(peak,Math.abs(x));const gain=options.normalize&&peak>0?Math.pow(10,-1/20)/peak:peak>1?1/peak:1;
 const introEdge=Math.round(rate*(plan.kind==='continuous-tail'?.1:.008)),outroEdge=Math.round(rate*.008);
 for(const c of result)for(let i=0;i<c.length;i++){let envelope=1;if(i<introEdge)envelope=i/introEdge;if(i>=c.length-outroEdge)envelope=Math.min(envelope,(c.length-1-i)/outroEdge);c[i]*=gain*envelope;}
 const mappedJoins=joins.map(t=>t*frames/length);
 return {channels:result,sampleRate:rate,duration:frames/rate,joins:mappedJoins,transitions,renderCheck:inspectRendered(result,rate,mappedJoins,options.bpm||120)};
}
function renderedBeatContinuity(channels,rate,join,bpm){
 const beat=60/bpm,stride=Math.max(1,Math.round(rate/11025)),sampleRate=rate/stride,size=1024,hop=256,start=Math.max(0,join-beat*3.8),end=Math.min(channels[0].length/rate,join+beat*3.8),frames=Math.max(0,Math.floor(((end-start)*sampleRate-size)/hop));
 if(frames<16)return {checked:false};
 const flux=new Float32Array(frames),previous=new Float32Array(size/2),real=new Float64Array(size),imag=new Float64Array(size),window=Float64Array.from({length:size},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(size-1)));
 for(let f=0;f<frames;f++){const at=Math.round(start*rate)+f*hop*stride;for(let k=0;k<size;k++){let sample=0;for(const channel of channels)sample+=(channel[at+k*stride]||0)/channels.length;real[k]=sample*window[k];imag[k]=0;}fft(real,imag);let onset=0;for(let k=2;k<size/2;k++){const magnitude=Math.log1p(Math.hypot(real[k],imag[k]));onset+=Math.max(0,magnitude-previous[k]);previous[k]=magnitude;}flux[f]=onset;}
 const onsetAt=t=>{const index=Math.round(((t-start)*sampleRate-size/2)/hop);let peak=0;for(let k=Math.max(0,index-1);k<=Math.min(frames-1,index+1);k++)peak=Math.max(peak,flux[k]);return peak;};
 const side=sign=>{const scores=[];for(let p=-16;p<=16;p++){const shift=p*beat/64;let score=0;for(let k=1;k<=3;k++){const t=join+sign*k*beat+shift;if(t>start+.05&&t<end-.05)score+=onsetAt(t);}scores.push(score);}const peak=Math.max(...scores),average=mean(scores),index=scores.indexOf(peak);return {phase:(index-16)/64,confidence:peak/(average+1e-6)};};
 const left=side(-1),right=side(1),drift=Math.abs(left.phase-right.phase);
 return {checked:left.confidence>=1.3&&right.confidence>=1.3,driftBeats:drift,leftConfidence:left.confidence,rightConfidence:right.confidence,passed:drift<=.2};
}
function inspectRendered(channels,rate,joins,bpm){
 const issues=[];let peak=0;
 for(const channel of channels)for(const sample of channel){if(!Number.isFinite(sample)){issues.push('音频样本异常');break;}peak=Math.max(peak,Math.abs(sample));}
 if(peak>1.0001)issues.push('成品削波');
 for(const time of joins){const pos=Math.round(time*rate),span=Math.round(rate*.2),near=Math.round(rate*.004);for(const channel of channels){const context=[];let click=0;for(let i=Math.max(1,pos-span);i<Math.min(channel.length,pos+span);i+=2){const delta=Math.abs(channel[i]-channel[i-1]);if(Math.abs(i-pos)<=near)click=Math.max(click,delta);else context.push(delta);}context.sort((a,b)=>a-b);const typical=context[Math.floor(context.length*.99)]||0;if(click>Math.max(.08,typical*6)){issues.push('接缝出现瞬态杂音');break;}}}
 const beatChecks=joins.map(t=>renderedBeatContinuity(channels,rate,t,bpm));
 if(beatChecks.some(x=>x.checked&&!x.passed))issues.push('接缝前后节拍相位不一致');
 return {passed:issues.length===0,issues:[...new Set(issues)],peak,beatChecks};
}
function wav(channels,sampleRate,markers=[]){const frames=channels[0].length,n=channels.length,cueSize=markers.length?12+24*markers.length:0,buffer=new ArrayBuffer(44+frames*n*2+cueSize),v=new DataView(buffer);const text=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};text(0,'RIFF');v.setUint32(4,buffer.byteLength-8,true);text(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,n,true);v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*n*2,true);v.setUint16(32,n*2,true);v.setUint16(34,16,true);text(36,'data');v.setUint32(40,frames*n*2,true);let offset=44;for(let i=0;i<frames;i++)for(let ch=0;ch<n;ch++){const s=clamp(channels[ch][i],-1,1);v.setInt16(offset,Math.round(s*(s<0?32768:32767)),true);offset+=2;}if(markers.length){text(offset,'cue ');v.setUint32(offset+4,4+markers.length*24,true);v.setUint32(offset+8,markers.length,true);offset+=12;markers.forEach((t,i)=>{v.setUint32(offset,i+1,true);v.setUint32(offset+4,Math.round(t*sampleRate),true);text(offset+8,'data');v.setUint32(offset+20,Math.round(t*sampleRate),true);offset+=24;});}return buffer;}
function remapLyrics(lyrics,plan){const result=[];let offset=0;const natural=planDuration(plan.pieces),scale=plan.duration/natural;for(const p of plan.pieces){for(const line of lyrics){const start=Math.max(p.start,line.start),end=Math.min(p.end,line.end);if(end>start)result.push({start:(offset+(start-p.start)/(p.rate||1))*scale,end:(offset+(end-p.start)/(p.rate||1))*scale,text:line.text});}offset+=(p.end-p.start)/(p.rate||1);}return result;}
const api={analyze,makePlans,mixPlans,assessPlans,render,stretch,wav,remapLyrics,planDuration,seamCost};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.WaveCutEngine=api;
})(typeof self!=='undefined'?self:globalThis);
