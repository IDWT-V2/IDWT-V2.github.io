const toc=document.querySelector('.page-toc');
if(toc && window.matchMedia('(max-width:760px)').matches)toc.open=false;
const form=document.querySelector('#library-search');
if(form){
  const query=document.querySelector('#query'),category=document.querySelector('#category'),topic=document.querySelector('#topic');
  const results=document.querySelector('#search-results'),status=document.querySelector('#search-status');
  let indexPromise,sequence=0;
  async function search(event){
    event?.preventDefault();const current=++sequence;
    const phrase=query.value.trim().toLowerCase();const words=phrase.split(/\s+/).filter(Boolean);
    const address=new URL(location.href);if(phrase)address.searchParams.set('q',query.value.trim());else address.searchParams.delete('q');history.replaceState(null,'',address);
    if(!words.length&&!category.value&&!topic.value){results.replaceChildren();status.textContent='Enter a phrase to search the theory and research record.';return;}
    status.textContent='Searching the library…';
    try{
      indexPromise??=fetch(form.dataset.index).then(response=>{if(!response.ok)throw new Error('Search index unavailable');return response.json();});
      const index=await indexPromise;if(current!==sequence)return;
      const found=index.filter(item=>(!category.value||item.category===category.value)&&(!topic.value||item.topic===topic.value)).map(item=>{
        const title=item.title.toLowerCase(),text=(item.title+' '+item.summary+' '+item.text).toLowerCase();
        return {...item,score:words.every(word=>text.includes(word))?words.reduce((sum,word)=>sum+(title.includes(word)?10:1),0): -1};
      }).filter(item=>item.score>=0).sort((a,b)=>b.score-a.score||a.title.localeCompare(b.title));
      results.replaceChildren();
      for(const item of found.slice(0,50)){
        const li=document.createElement('li'),label=document.createElement('span'),link=document.createElement('a'),summary=document.createElement('p');
        label.className='result-category';label.textContent=item.category+(item.topic?' / '+item.topic:'');
        link.href=item.url;link.textContent=item.title;summary.textContent=item.summary;
        li.append(label,link,summary);results.append(li);
      }
      status.textContent=found.length?`${found.length} result${found.length===1?'':'s'}${found.length>50?' · showing the first 50':''}.`:'No matching pages. Try a broader phrase or another topic.';
    }catch(error){indexPromise=undefined;status.textContent='Search could not load. Please try again, or use the Theory and Research indexes.';}
  }
  form.addEventListener('submit',search);category.addEventListener('change',search);topic.addEventListener('change',search);
  let timer;query.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(search,180);});
  query.value=new URL(location.href).searchParams.get('q')??'';if(query.value)search();
}
