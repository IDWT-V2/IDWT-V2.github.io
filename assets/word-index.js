export function normalize_word(value) {
  return value.normalize('NFKC').toLowerCase();
}

export function word_tokens(value) {
  const text=normalize_word(value).replace(/(?<![\p{L}\p{N}])[-+]?(?:\d+\.?\d*|\.\d+)e[-+]?\d+(?![\p{L}\p{N}])/gu,' ');
  return [...new Set((text.match(/[\p{L}\p{N}]+/gu)??[]).filter(word=>/\p{L}/u.test(word)&&!/^[a-f0-9]{32,}$/.test(word)))];
}

export function word_index_group(value) {
  const prefix=word_tokens(value)[0];
  if(!prefix)return null;
  return /^[a-z]/.test(prefix)?prefix[0].toUpperCase():/^\p{N}/u.test(prefix)?'#':Array.from(prefix)[0].toUpperCase();
}

export function word_matches(word, query) {
  const prefix=word_tokens(query)[0];
  return !prefix||word.startsWith(prefix);
}
