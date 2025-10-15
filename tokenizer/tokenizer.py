import regex as re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Iterable, Iterator
import os
import json
from tqdm import tqdm
import time
from utils.common import gpt2_bytes_to_unicode
class Tokenizer:
    """
    Construct a BPE Tokenizer from scratch.
    """
    vocab: Dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: Dict[str, bytes]  #  实例变量类型注释在这里
    next_token_id: int
    pat: re.Pattern[str]
    def __init__(self, vocab: Dict[int, bytes] | None, merges: List[Tuple[bytes, bytes]] | None, special_tokens: List[str] | None = None):
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens
        """
        self.vocab = vocab if vocab else {}
        self.merges = merges if merges else []
        self.special_tokens = {token: token.encode('utf-8') for token in special_tokens} if special_tokens else {}
        
        self.next_token_id = max(vocab.keys()) + 1 if vocab else 0
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def get_vocab_merges_from_path(
        cls, 
        vocab_path: str | os.PathLike, 
        merges_path: str | os.PathLike, 
        special_tokens: List[str] | None = None
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Class method that reads a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) from the given paths
        and returns them as a tuple (vocab, merges).
        """
        gpt2_byte_decoder: dict[str, int] = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab: dict[str, int] = json.load(vocab_f)
        gpt2_bpe_merges: list[tuple[str, str]] = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    a, b = cleaned_line.split(" ")
                    gpt2_bpe_merges.append((a, b))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return (vocab, merges)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: List[str] | None = None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        vocab, merges = cls.get_vocab_merges_from_path(vocab_path, merges_path, special_tokens)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _initialize_vocab(self):
        """Initialize vocabulary with bytes and special tokens"""
        # Add special tokens
        for token_bytes in sorted(self.special_tokens.values(), reverse=True):
            self.vocab[self.next_token_id] = token_bytes
            self.next_token_id += 1
        # Add bytes 0-255
        for i in range(256):
            self.vocab[self.next_token_id] = bytes([i])
            self.next_token_id += 1
        

    
    def _pretokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using the GPT-2 style regex pattern
        Returns: list of pre-tokens (strings)
        """
        # Use regex to find all pre-tokens
        pre_tokens = self.pat.findall(text)
        return [token for token in pre_tokens ]  # Don't remove empty tokens! otherwise spaces are lost
    
    def _words_to_byte_tuples(self, pre_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
        """
        Convert pre-tokens to byte tuples and count frequencies
        """
        word_freq = Counter()
        
        for token in pre_tokens:
            # Convert each pre-token to bytes and split into individual bytes
            byte_sequence = token.encode('utf-8')
            byte_tuple = tuple(bytes([b]) for b in byte_sequence)
            word_freq[byte_tuple] += 1
            
        return dict(word_freq)
    
    def _get_stats(self, word_freq: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
        """Get frequency of all adjacent pairs in the vocabulary"""
        pairs:dict[Tuple[bytes,bytes], int] = defaultdict(int)
        
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
                
        return dict(pairs)

    def _find_pair(self, pairs: dict[tuple[bytes, bytes], int], to_find:tuple[bytes,bytes]):
        print(f"fuckkkkk!!!!!!!!!!!\nto_find {to_find}")
        for pair, freq in pairs.items():
            if pair[0] == "\n".encode():
                print(f"pair,{pair}")
            # print(f"pair[0],{pair[0]},/n.enocode(), {"\n".encode()}")
            if pair == to_find:
                print(f"fuckkkkk!!!!!!!!!!!\npair {pair}, freq {freq}")
                break

    def _find_max_pair(self, pairs: Dict[Tuple[bytes, bytes], int]) -> Optional[Tuple[bytes, bytes]]:
        """Find the most frequent pair, breaking ties lexicographically"""
        if not pairs:
            return None
            
        max_freq = max(pairs.values())
        # Get all pairs with max frequency
        candidates = [pair for pair, freq in pairs.items() if freq == max_freq]
        
        # Break tie by choosing lexicographically greater pair
        return max(candidates)
    
    def _merge_pair(self, pair: Tuple[bytes, bytes], word_freq: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
        """Merge the given pair in all words, return a merged word_freq"""
        a, b = pair
        merged = a + b  # Concatenate bytes
        
        new_word_freq = {}
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq
            
        return new_word_freq
    
    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Train BPE tokenizer
        
        Args:
            input_path: path to training text file
            vocab_size: desired vocabulary size (including special tokens)
            special_tokens: list of special tokens
            
        Returns:
            vocab: token_id -> bytes
            merges: list of merge operations
        """
        # Read training data
        input_path = os.fspath(input_path)
        # print(f"inpu_path,{input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Store special tokens as bytes.  dict[str, bytes]
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens = {token: token.encode('utf-8') for token in special_tokens} 
        
    
        # Remove special tokens from training data and split into documents
        start_time = time.time()
        documents:list[str] = self._split_by_special_tokens(text, special_tokens)
        # print("documents[:9},",documents[:8])
        end_time = time.time()
        print(f"Removed special tokens and split into {len(documents)} documents in {end_time - start_time:.2f} seconds")
        
        # Initialize vocabulary
        self._initialize_vocab()
        
        # Pre-tokenize each document and 
        start_time = time.time()
        word_freq:dict[Tuple[bytes,...], int] = defaultdict(int)
        for doc in documents:
            # Step 1: Pre-tokenization with regex
            pre_tokens:list[str] = self._pretokenize(doc)
    
            # Step 2: Convert pre-tokens to byte tuples
            doc_word_freq = self._words_to_byte_tuples(pre_tokens)
            
            for word, freq in doc_word_freq.items():
                word_freq[word] += freq
        end_time = time.time()
        print(f"Pre-tokenized in {end_time - start_time:.2f} seconds")

        # aggregate frequencies
        start_time = time.time()
        pairs = self._get_stats(word_freq)
        end_time = time.time()
        print(f"Aggregated frequencies in {end_time - start_time:.2f} seconds")

        # print(f"Initial vocabulary size: {len(self.vocab)}")
        # print(f"Number of unique pre-tokens: {len(word_freq)}")
        
        # Perform BPE merges until we reach desired vocab size
        with tqdm(total=vocab_size - len(self.vocab), ncols=80, desc="Merging pairs") as pbar:
            while len(self.vocab) < vocab_size:
                # Get statistics on pairs
                pairs = self._get_stats(word_freq)
                if not pairs:
                    print("No more pairs to merge")
                    break
                    
                # Find the most frequent pair
                best_pair = self._find_max_pair(pairs)
                if best_pair is None:
                    break
                    
                # Merge this pair
                word_freq = self._merge_pair(best_pair, word_freq)
                
                # Add new token to vocabulary
                new_token = best_pair[0] + best_pair[1]
                self.vocab[self.next_token_id] = new_token
                self.next_token_id += 1
                
                # Record this merge
                self.merges.append(best_pair)

                # update pbar
                pbar.update(1)
                # if len(self.vocab) % 100 == 0:
                #   print(f"Merged {best_pair} -> {new_token}, vocab size: {len(self.vocab)}")
        
        # print(f"Final vocabulary size: {len(self.vocab)}")
        # print(f"Total merges: {len(self.merges)}")
        
        return self.vocab, self.merges
    
    


    def save(self, vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
        """
        Save the vocabulary and merges to files
        But json can't save bytes, this is a problem
        vocab_path: path to save vocabulary (JSON format)
        merges_path: path to save merges (text format)
        """
        gpt2_int_str_encoder:dict[int, str] = gpt2_bytes_to_unicode()
        vocab_serialized: dict[str, int] = { "".join([gpt2_int_str_encoder[byte] for byte in v]): k for k, v in self.vocab.items()}
        
        merges_serialized: list[Tuple[str, str]] = [
            ("".join([gpt2_int_str_encoder[byte] for byte in a]), "".join([gpt2_int_str_encoder[byte] for byte in b]))
            for a, b in self.merges
        ]
        # open file 
        with open(os.fspath(vocab_path), 'w', encoding='utf-8') as vocab_f:
            json.dump(vocab_serialized, vocab_f, ensure_ascii=False, indent=2)
        with open(os.fspath(merges_path), 'w', encoding='utf-8') as merges_f:
            for a, b in merges_serialized:
                merges_f.write(f"{a} {b}\n")

    def from_pretrained(self):
        """
        Implemented in from_files. Use that method instead.
        """
        pass


    def _split_by_special_tokens(self, text: str, special_tokens: List[str]) -> list[str]:
        """
        split by special tokens
        Returns a list of documents and special tokens as separate entries
        """
        if not special_tokens:
            return [text]
            
        # Escape special tokens for regex
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, reverse=True)]
        pattern = "|".join(escaped_tokens)
        
        # Split on special tokens
        parts = re.split(f"({pattern})", text)
        # Return only the non-special token parts (documents)
        documents_or_special_token = []
        special_tokens_found = []
        for i, part in enumerate(parts):
            documents_or_special_token.append(part)

        return [doc for doc in documents_or_special_token if doc]  # Remove empty documents

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using trained BPE
        """
        # Remove special tokens and pre-tokenize
        document_or_special_tokens = self._split_by_special_tokens(text, list(self.special_tokens.keys()))
        
        all_token_ids: list[int] = []

        for doc_or_special_token in (document_or_special_tokens):
            # if not doc_or_special_token.strip():
            #     continue
                
            # Pre-tokenize the document
            # If it's a special token, just add it directly
            if doc_or_special_token in self.special_tokens:
                special_token_bytes = self.special_tokens[doc_or_special_token]
                # Find token ID in vocab
                token_id = None
                for tid, vocab_token in self.vocab.items():
                    if vocab_token == special_token_bytes:
                        token_id = tid
                        break
                if token_id is not None:
                    all_token_ids.append(token_id)
                continue
            
            # Not a special token, pre-tokenize normally
            pre_tokens: list[str] = self._pretokenize(doc_or_special_token)

            for pre_token in pre_tokens:
                # Convert pre-token to bytes
                byte_sequence = pre_token.encode('utf-8')
                current_tokens = [bytes([b]) for b in byte_sequence]
                
                # Apply BPE merges
                for merge in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(current_tokens):
                        if (i < len(current_tokens) - 1 and 
                            current_tokens[i] == merge[0] and 
                            current_tokens[i + 1] == merge[1]):
                            new_tokens.append(merge[0] + merge[1])
                            i += 2
                        else:
                            new_tokens.append(current_tokens[i])
                            i += 1
                    current_tokens = new_tokens
                
                # Convert tokens to token IDs
                for token in current_tokens:
                    # Find token ID in vocab
                    token_id = None
                    for tid, vocab_token in self.vocab.items():
                        if vocab_token == token:
                            token_id = tid
                            break
                    
                    if token_id is not None:
                        all_token_ids.append(token_id)
                    else:
                        # Fallback: split into bytes
                        for byte in token:
                            for tid, vocab_token in self.vocab.items():
                                if vocab_token == bytes([byte]):
                                    all_token_ids.append(tid)
                                    break
            # Add special tokens found in the original text.just one per doc
            

        return all_token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        
        Memory considerations. Suppose we want to tokenize a large text file that we cannot fit in memory.
        To eﬀiciently tokenize this large file (or any other stream of data), we need to break it up into manageable
        chunks and process each chunk in-turn, so that the memory complexity is constant as opposed to linear in
        the size of the text. In doing so, we need to make sure that a token doesn’t cross chunk boundaries, else
        we’ll get a different tokenization than the naïve method of tokenizing the entire sequence in-memory
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id
        
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        """
        tokens = [self.vocab.get(token_id, b'') for token_id in token_ids]
        # Concatenate bytes and decode to string
        result_bytes = b''.join(tokens)
        
        try:
            return result_bytes.decode('utf-8', errors='replace')
        except:
            return result_bytes.decode('utf-8', errors='ignore')
