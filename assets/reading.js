import {word_matches} from './word-index.js';

const toc=document.querySelector('.page-toc');
if(toc && window.matchMedia('(max-width:760px)').matches)toc.open=false;
const form=document.querySelector('#library-search');
if(form){
  const query=document.querySelector('#query'),category=document.querySelector('#category'),topic=document.querySelector('#topic');
  const status=document.querySelector('#search-status');
  const letters=document.querySelector('.WordIndexLetters');
  const pages=JSON.parse(document.querySelector('#word-index-pages').textContent);
  const groups=[...document.querySelectorAll('.WordIndexGroup')].map(group=>({element:group,entries:[...group.querySelectorAll('.WordIndexEntry')]}));
  let had_filters=false;
  function search(event){
    event?.preventDefault();
    const phrase=query.value.trim();
    const address=new URL(location.href);
    if(phrase)address.searchParams.set('q',phrase);else address.searchParams.delete('q');
    history.replaceState(null,'',address);
    const filtered=Boolean(category.value||topic.value);
    letters.hidden=Boolean(phrase||filtered);
    const allowed=pages.map(page=>(!category.value||page.category===category.value)&&(!topic.value||page.topic===topic.value));
    let word_count=0;
    for(const group of groups){
      let group_count=0;
      for(const entry of group.entries){
        let visible=word_matches(entry.dataset.word,phrase);
        if(visible&&(filtered||had_filters)){
          const links=entry.querySelectorAll('a');
          visible=false;
          for(const link of links){link.hidden=!allowed[Number(link.dataset.page)];if(!link.hidden)visible=true;}
        }
        entry.hidden=!visible;
        if(visible){word_count++;group_count++;}
      }
      group.element.hidden=group_count===0;
    }
    // Hidden entries may still contain filtered links when a query changes.
    if(filtered)had_filters=true;
    status.textContent=word_count?`${word_count.toLocaleString()} matching word${word_count===1?'':'s'}.${phrase||filtered?'':' Showing the complete index.'}`:'No matching words. Try another word or part of a word.';
  }
  form.addEventListener('submit',search);category.addEventListener('change',search);topic.addEventListener('change',search);
  let timer;query.addEventListener('input',()=>{
    clearTimeout(timer);
    if(!query.value.trim()){category.value='';topic.value='';}
    timer=setTimeout(search,180);
  });
  query.value=new URL(location.href).searchParams.get('q')??'';if(query.value)search();
}
