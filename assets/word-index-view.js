import {word_index_group,word_matches} from './word-index.js';

export function select_index_entries(entries,pages,{query='',category='',topic=''}={}) {
    return entries.filter(entry=>word_matches(entry.word,query)).map(entry=>({word:entry.word,pages:entry.pages.filter(id=>(!category||pages[id].category===category)&&(!topic||pages[id].topic===topic))})).filter(entry=>entry.pages.length);
}

export class WordIndexView {
    static async init() {
        const view=new WordIndexView(document.querySelector('.search-page'));
        await view.start();
        return view;
    }

    constructor(root) {
        this.root=root;
        this.form=root.querySelector('#library-search');
        this.query=root.querySelector('#query');
        this.category=root.querySelector('#category');
        this.topic=root.querySelector('#topic');
        this.status=root.querySelector('#search-status');
        this.letters=root.querySelector('#word-index-letters');
        this.results=root.querySelector('#search-results');
        this.pagination=root.querySelector('#word-index-pagination');
        this.cache=new Map();
        this.offset=0;
        this.pageSize=100;
        this.revision=0;
        this.letter=null;
        this.form.addEventListener('submit',event=>{event.preventDefault();this.offset=0;this.update();});
        for(const input of [this.category,this.topic])input.addEventListener('change',()=>{this.offset=0;this.update();});
        this.query.addEventListener('input',()=>{
            clearTimeout(this.timer);
            this.revision++;
            this.offset=0;
            if(!this.query.value.trim()){this.category.value='';this.topic.value='';this.letter=null;}
            this.timer=setTimeout(()=>this.update(),180);
        });
        root.querySelector('#clear-word-index').addEventListener('click',()=>this.clear());
        this.letters.addEventListener('click',event=>{
            const link=event.target.closest('[data-word-index-letter]');
            if(!link||!this.manifest)return;
            event.preventDefault();
            this.letter=this.manifest.groups.find(group=>group.letter===link.dataset.wordIndexLetter);
            this.query.value='';this.offset=0;this.update();
        });
    }

    async json(url) {
        const response=await fetch(url);
        if(!response.ok)throw new Error(`Word index request failed: ${response.status}`);
        return response.json();
    }

    async start() {
        this.manifest=await this.json(this.root.dataset.wordIndexManifest);
        if(!Array.isArray(this.manifest.groups))throw new Error('Invalid word index manifest');
        this.query.value=new URL(location.href).searchParams.get('q')??'';
        this.letter=this.manifest.groups.find(group=>group.id===location.hash.slice(1))??null;
        await this.update();
    }

    getPages() {
        if(!this.pagesPromise)this.pagesPromise=this.json(this.manifest.pages_url).then(pages=>{
            if(!Array.isArray(pages)||pages.length!==this.manifest.page_count)throw new Error('Invalid word index page metadata');
            return pages;
        });
        return this.pagesPromise;
    }

    getLetter(group) {
        if(!this.cache.has(group.url))this.cache.set(group.url,this.json(group.url).then(data=>{
            if(data.letter!==group.letter||data.entries.length!==group.word_count)throw new Error('Invalid word index letter data');
            return data.entries;
        }));
        return this.cache.get(group.url);
    }

    clear() {
        clearTimeout(this.timer);
        this.query.value='';this.category.value='';this.topic.value='';this.letter=null;this.offset=0;
        this.update();
    }

    async update() {
        if(!this.manifest)return;
        const revision=++this.revision;
        const query=this.query.value.trim();
        const address=new URL(location.href);
        if(query){address.searchParams.set('q',query);address.hash='';}
        else{address.searchParams.delete('q');address.hash=this.letter?.id??'';}
        history.replaceState(null,'',address);
        this.letters.hidden=Boolean(query);
        for(const link of this.letters.querySelectorAll('a')){
            if(!query&&link.dataset.wordIndexLetter===this.letter?.letter)link.setAttribute('aria-current','true');
            else link.removeAttribute('aria-current');
        }
        if(!query&&!this.letter){
            this.results.replaceChildren();this.pagination.replaceChildren();
            this.status.textContent=`${this.manifest.word_count.toLocaleString()} words · ${this.manifest.link_count.toLocaleString()} page links. Choose a letter or search by word prefix.`;
            return;
        }
        this.status.textContent='Loading matching words…';
        try{
            const group=query?this.manifest.groups.find(item=>item.letter===word_index_group(query)):this.letter;
            const entries_request=query?(group?this.getLetter(group):Promise.resolve([])):this.getLetter(group);
            const [entries,pages]=await Promise.all([entries_request,this.getPages()]);
            if(revision!==this.revision)return;
            this.pages=pages;
            const selected=select_index_entries(entries,pages,{query,category:this.category.value,topic:this.topic.value});
            this.render(selected);
        }catch(error){
            if(revision!==this.revision)return;
            this.results.replaceChildren();this.pagination.replaceChildren();
            this.status.textContent='The word index could not be loaded. Please reload the page.';
            console.error(error);
        }
    }

    createEntry(entry) {
        const row=document.createElement('div');row.className='WordIndexEntry';
        const heading=document.createElement('h3');heading.textContent=entry.word;row.append(heading);
        const links=document.createElement('div');links.className='WordIndexLinks';row.append(links);
        const append=ids=>{
            const fragment=document.createDocumentFragment();
            for(const id of ids){const page=this.pages[id],link=document.createElement('a');link.href=page.url;link.textContent=page.title;fragment.append(link);}
            links.append(fragment);
        };
        append(entry.pages.slice(0,12));
        if(entry.pages.length>12){
            const more=document.createElement('button');more.type='button';more.className='button WordIndexMoreButton';more.textContent=`Show All ${entry.pages.length.toLocaleString()} Pages`;
            more.addEventListener('click',()=>{append(entry.pages.slice(12));more.remove();});row.append(more);
        }
        return row;
    }

    render(entries) {
        const fragment=document.createDocumentFragment();
        for(const entry of entries.slice(this.offset,this.offset+this.pageSize))fragment.append(this.createEntry(entry));
        this.results.replaceChildren(fragment);this.pagination.replaceChildren();
        const prefix=this.query.value.trim()?'':`Letter ${this.letter.letter}: `;
        if(!entries.length){this.status.textContent=prefix+'No matching words. Try another prefix.';return;}
        const end=Math.min(this.offset+this.pageSize,entries.length);
        this.status.textContent=`${prefix}${entries.length.toLocaleString()} matching word${entries.length===1?'':'s'}. Showing ${(this.offset+1).toLocaleString()}–${end.toLocaleString()}.`;
        for(const [id,label,offset,disabled] of [['word-index-previous','Previous Words',Math.max(0,this.offset-this.pageSize),this.offset===0],['word-index-next','Next Words',this.offset+this.pageSize,end===entries.length]]){
            const button=document.createElement('button');button.type='button';button.id=id;button.className='button';button.textContent=label;button.disabled=disabled;
            button.addEventListener('click',()=>{this.offset=offset;this.render(entries);this.results.scrollIntoView({block:'start',behavior:'instant'});});this.pagination.append(button);
        }
    }
}
