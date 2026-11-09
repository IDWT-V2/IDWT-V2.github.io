const toc=document.querySelector('.page-toc');
if(toc && window.matchMedia('(max-width:760px)').matches)toc.open=false;
if(document.querySelector('#library-search')){
  import('./word-index-view.js').then(({WordIndexView})=>WordIndexView.init()).catch(error=>{
    const status=document.querySelector('#search-status');
    status.textContent='The word index could not be loaded. Please reload the page.';
    status.setAttribute('role','alert');
    console.error(error);
  });
}
