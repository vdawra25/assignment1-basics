import regex as re
from sortedcontainers import SortedList
from functools import total_ordering
from multiprocessing import Pool
import time
import os
import pprint
# import tracemalloc
@total_ordering
class Item:
    def __init__(self, num, pair):
        self.num = num
        self.pair = pair

    def __eq__(self, other):
        return (self.num, self.pair) == (other.num, other.pair)

    def __lt__(self, other):
        # reversed comparison -> descending order
        return (self.num, self.pair) > (other.num, other.pair)

    def __repr__(self):
        return f"({self.num}, {self.pair})"


def tokenize_chunk(outputs):
    map1 = dict()
    map2 = dict()
    # print("output length =>", len(outputs))
    for i,   output in enumerate(outputs):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, output)
        for pre_token in pre_tokens:
            pre_token_bytes = pre_token.encode("utf-8")
            if pre_token_bytes in map2:
                map2[pre_token_bytes]+=1
            else:
                map2[pre_token_bytes] = 1    
            for k in range(len(pre_token_bytes)):
                if k == len(pre_token_bytes) - 1:
                    break
                toLookup = (bytes([pre_token_bytes[k]]), bytes([pre_token_bytes[k+1]]))
                if toLookup in map1:
                    map1[toLookup]+=1
                else:
                    map1[toLookup] = 1

    return map1, map2                

def merge_partials(partials):
    map1 = {}
    map2 = {}
    sorted = SortedList()
    for partial in partials:
        mp1 = partial[0]
        mp2 = partial[1]
        for k, v in mp1.items():
            if k in map1:
                map1[k]+=v
            else:
                map1[k] = v    
        for k, v in mp2.items():
            if k in map2:
                map2[k]+=v
            else:
                map2[k]=v
    for k , v in map1.items():
        sorted.add(Item(v, k))
    return map1, map2, sorted    


def merge_bp(map1, map2, sorted,vocab_size,speical_tokens):
    merges = list()
    count = 0
    merge_start = time.time()
    while 256 + len(merges) + len(speical_tokens) < vocab_size:
        count+=1
        item = sorted[0]
        toLookup  = pair = item.pair
        token1 = pair[0]
        token2 = pair[1]          
        merges.append(toLookup)
        # if count%100 == 0:
        #     print("merge done")
        changed = dict()
        changed2 = dict()
        merged = token1 + token2
        for key, cnt in map2.items():
            if isinstance(key, tuple):
                word = key
            else:
                word = tuple(bytes([b]) for b in key)
            new_word = list()
            found = False
            k = 0
            while k < len(word):
                if not (k + 1 < len(word) and word[k] == token1 and word[k + 1] == token2):
                    new_word.append(word[k])
                    k += 1
                    continue
                found = True
                if toLookup not in changed:
                    changed[toLookup] = [map1[toLookup], None]
                map1[toLookup] -= cnt
                changed[toLookup][1] = map1[toLookup]

                # backward neighbour: (prev, token1) -> (prev, merged)
                if new_word:
                    prev = new_word[-1]
                    toRemove = (prev, token1)
                    toAdd = (prev, merged)
                    if toRemove not in changed:
                        changed[toRemove] = [map1[toRemove], None]
                    map1[toRemove] -= cnt
                    changed[toRemove][1] = map1[toRemove]
                    if toAdd not in changed:
                        changed[toAdd] = [map1.get(toAdd, 0), None]
                    map1[toAdd] = map1.get(toAdd, 0) + cnt
                    changed[toAdd][1] = map1[toAdd]

                # forward neighbour: (token2, next) -> (merged, next)
                if k + 2 < len(word):
                    nxt = word[k + 2]
                    toRemove = (token2, nxt)
                    toAdd = (merged, nxt)
                    if toRemove not in changed:
                        changed[toRemove] = [map1[toRemove], None]
                    map1[toRemove] -= cnt
                    changed[toRemove][1] = map1[toRemove]
                    if toAdd not in changed:
                        changed[toAdd] = [map1.get(toAdd, 0), None]
                    map1[toAdd] = map1.get(toAdd, 0) + cnt
                    changed[toAdd][1] = map1[toAdd]

                new_word.append(merged)
                k += 2
            if found:
                changed2[key] = tuple(new_word)

        for old_key, new_key in changed2.items():
            map2[new_key] = map2.pop(old_key)

        for k, (old, new) in changed.items():
            sorted.discard(Item(old, k))
            sorted.add(Item(new, k))
        del map1[toLookup]
        sorted.discard(Item(0, pair))

    merge_end = time.time()
    print("merge completed in =>", merge_end - merge_start)
    #output 1 => dict [tokenID, bytes] (0, 255 => direct), utf-8 encoding for rest of them
    op1 = dict()
    for i in range(256):
        op1[i] = bytes([i])

    cnt = 256
    for merge in merges:
        token1 = merge[0]
        token2 = merge[1]
        s = token1 + token2
        op1 [cnt] = s
        cnt+=1

    for spe in speical_tokens:
        op1[cnt] = spe.encode("utf-8")
        cnt+=1
    result = [(a, b) for a, b in merges]  
    output = {
    "vocab_keys": set(op1.keys()),
    "vocab_values": set(op1.values()),
    "merges": merges,
    }


    with open("tests/_snapshots/tiny_stories_full.txt", "w") as f:
        pprint.pprint(output, stream=f)
    return op1,result


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    speical_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    filePath = input_path 
    data = ""
    with open(filePath, "r") as f:
        data = f.read()
    outputs = re.split("|".join(re.escape(token) for token in speical_tokens),data)
    del data
    numProcess = 8
    chunkSize = -(-len(outputs) // numProcess)  # ceiling division
    chunks = [outputs[i:i + chunkSize] for i in range(0, len(outputs), chunkSize)]
    start = time.time()
    partials = []
    with Pool(processes=numProcess) as pool: 
        partials = pool.map(tokenize_chunk, chunks)
    end = time.time()
    print("pre-tokenize done in =>", end-start) 
    map1, map2, sorted = merge_partials(partials)
    end1 = time.time()
    print("partials merged in =>", end1 - end) 
    return merge_bp(map1, map2, sorted, vocab_size,speical_tokens)

    


if __name__ == "__main__":
    train_bpe("data/TinyStoriesV2-GPT4-train.txt",10000,  ["<|endoftext|>"])
    # filePath = "data/TinyStoriesV2-GPT4-train.txt" 
    # data = ""
    # with open(filePath, "r") as f:
    #     data = f.read()
    # outputs = re.split("|".join(re.escape(token) for token in ["<|endoftext|>"]),data)
    # del data
    # numProcess = 8
    # chunkSize = -(-len(outputs) // numProcess)  # ceiling division
    # chunks = [outputs[i:i + chunkSize] for i in range(0, len(outputs), chunkSize)]
    # start = time.time()
    # partials = []
    # with Pool(processes=8) as pool: 
    #     partials = pool.map(tokenize_chunk, outputs)
    # end = time.time()
    # print("pre-tokenize done in =>", end-start) 
    # map1, map2, sorted = merge_partials(partials)
    # end1 = time.time()
    # print("partials merged in =>", end1 - end) 

