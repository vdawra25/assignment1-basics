import regex as re
from sortedcontainers import SortedList
from functools import total_ordering
from multiprocessing import Pool
import time
import os
import pprint
import cProfile
import pstats
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
    map2 = dict()
    for output in outputs:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, output)
        for pre_token in pre_tokens:
            if pre_token in map2:
                map2[pre_token]+=1
            else:
                map2[pre_token]=1

    return map2                

def merge_partials(partials):
    map1 = {}     # byte-pair => freq map
    map2 = {}     # pre-token => freq map
    pre_token_bytes_list = []     # list of pre-token-bytes
    sorted = SortedList()
    for partial in partials:
        mp2 = partial
        for k, v in mp2.items():
            new_key = k.encode("utf-8")
            if new_key in map2:
                map2[new_key]+=v
            else:
                map2[new_key]=v

    pre_token_bytes_list = list(map2.keys())
    # map1[pair] = (count, {idx in pre_token_bytes_list: occurrences of pair in that pre-token})
    for idx, pre_token_bytes in enumerate(pre_token_bytes_list):
        v = map2[pre_token_bytes]
        for k in range(len(pre_token_bytes)):
                if k == len(pre_token_bytes) - 1:
                    break
                toLookup = (bytes([pre_token_bytes[k]]), bytes([pre_token_bytes[k+1]]))
                if toLookup in map1:
                    idx_map = map1[toLookup][1]
                    idx_map[idx] = idx_map.get(idx, 0) + 1
                    map1[toLookup] = (map1[toLookup][0] + v, idx_map)
                else:
                    map1[toLookup] = (v, {idx: 1})
    for k , v in map1.items():
        sorted.add(Item(v[0], k))
    print("pre-token-count", len(map2))
    print("byte-pair-count", len(map1))
    return map1,map2,pre_token_bytes_list,sorted    


def merge_bp(map1,map2, pre_token_bytes_list ,sorted,vocab_size,speical_tokens):
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
        changed = dict()
        changed2 = dict()
        merged = token1 + token2
        idx_list = map1[toLookup][1]
        for idx in idx_list:
            key = pre_token_bytes_list[idx]
            cnt = map2[key]
            if isinstance(key, tuple):
                word = key
            else:
                word = tuple(bytes([b]) for b in key)
            x = len(word)
            new_word = [None] * x
            j = 0
            found = False
            k = 0
            while k < x:
                if not (k + 1 < x and word[k] == token1 and word[k + 1] == token2):
                    new_word[j] = word[k]
                    j += 1
                    k += 1
                    continue
                found = True
                # (k, k+1): the merged pair itself
                if toLookup not in changed:
                    changed[toLookup] = [map1[toLookup][0], None]
                map1[toLookup] = (map1[toLookup][0] - cnt, map1[toLookup][1])
                map1[toLookup][1][idx] -= 1
                changed[toLookup][1] = map1[toLookup][0]

                # backward neighbour: (prev, token1) -> (prev, merged)
                if j > 0:
                    prev = new_word[j - 1]
                    toRemove = (prev, token1)
                    toAdd = (prev, merged)
                    if toRemove not in changed:
                        changed[toRemove] = [map1[toRemove][0], None]
                    map1[toRemove] = (map1[toRemove][0] - cnt, map1[toRemove][1])
                    map1[toRemove][1][idx] -= 1
                    changed[toRemove][1] = map1[toRemove][0]
                    if toAdd not in changed:
                        changed[toAdd] = [map1[toAdd][0] if toAdd in map1 else 0, None]
                    if toAdd in map1:
                        map1[toAdd] = (map1[toAdd][0] + cnt, map1[toAdd][1])
                        map1[toAdd][1][idx] = map1[toAdd][1].get(idx, 0) + 1
                    else:
                        map1[toAdd] = (cnt, {idx: 1})
                    changed[toAdd][1] = map1[toAdd][0]

                # forward neighbour: (token2, next) -> (merged, next)
                if k + 2 < x:
                    nxt = word[k + 2]
                    toRemove = (token2, nxt)
                    toAdd = (merged, nxt)
                    if toRemove not in changed:
                        changed[toRemove] = [map1[toRemove][0], None]
                    map1[toRemove] = (map1[toRemove][0] - cnt, map1[toRemove][1])
                    map1[toRemove][1][idx] -= 1
                    changed[toRemove][1] = map1[toRemove][0]
                    if toAdd not in changed:
                        changed[toAdd] = [map1[toAdd][0] if toAdd in map1 else 0, None]
                    if toAdd in map1:
                        map1[toAdd] = (map1[toAdd][0] + cnt, map1[toAdd][1])
                        map1[toAdd][1][idx] = map1[toAdd][1].get(idx, 0) + 1
                    else:
                        map1[toAdd] = (cnt, {idx: 1})
                    changed[toAdd][1] = map1[toAdd][0]

                new_word[j] = merged
                j += 1
                k += 2
            if found:
                changed2[key] = tuple(new_word[:j])
                pre_token_bytes_list[idx] = changed2[key]

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
):
    filePath = input_path 
    data = ""
    with open(filePath, "r") as f:
        data = f.read()
    outputs = re.split("|".join(re.escape(token) for token in speical_tokens),data)
    del data
    numProcess = 8
    chunkSize = -(-len(outputs) // numProcess)  # ceiling division
    chunkStartTime = time.time()
    chunks = [outputs[i:i + chunkSize] for i in range(0, len(outputs), chunkSize)]
    start = time.time()
    print("chunking done in =>", start-chunkStartTime)
    partials = []
    with Pool(processes=numProcess) as pool: 
        partials = pool.map(tokenize_chunk, chunks)
    end = time.time()
    print("pre-tokenize done in =>", end-start) 
    map1,map2,pre_tokens_bytes_list ,sorted = merge_partials(partials)
    end1 = time.time()
    print("partials merged in =>", end1 - end) 
    # with cProfile.Profile() as mergeProfiler:
    #     result = merge_bp(map1,map2,pre_tokens_bytes_list, sorted ,vocab_size,speical_tokens)
    # mergeStats = pstats.Stats(mergeProfiler).strip_dirs()
    # mergeStats.sort_stats("cumtime").print_stats(30)
    # mergeStats.sort_stats("tottime").print_stats(30)
    return merge_bp(map1,map2,pre_tokens_bytes_list, sorted ,vocab_size,speical_tokens)

    


if __name__ == "__main__":
    start = time.time()
    train_bpe("data/TinyStoriesV2-GPT4-train.txt",10000,  ["<|endoftext|>"])
    end = time.time()
    print("done in =>", end-start)
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

