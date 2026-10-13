export function normalize_word(value) {
  return value.normalize('NFKC').toLowerCase();
}

export function word_tokens(value) {
  const text=normalize_word(value).replace(/(?<![\p{L}\p{N}])[-+]?(?:\d+\.?\d*|\.\d+)e[-+]?\d+(?![\p{L}\p{N}])/gu,' ');
  return [...new Set((text.match(/[\p{L}\p{N}]+/gu)??[]).filter(word=>/\p{L}/u.test(word)&&!/^[a-f0-9]{32,}$/.test(word)))];
}

export function word_matches(word, query) {
  return word.includes(normalize_word(query.trim()));
}
