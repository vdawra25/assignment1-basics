import regex as re
from sortedcontainers import SortedList
from functools import total_ordering
import os
import time
import cProfile
import pstats
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


# placeholder written at the slot of the right-hand token after a merge
SKIP = "<|endoftext|>"


def next_idx(word, k):
    """index of first non-SKIP element after k (len(word) if none)"""
    k += 1
    while k < len(word) and word[k] == SKIP:
        k += 1
    return k


def prev_idx(word, k):
    """index of first non-SKIP element before k (-1 if none)"""
    k -= 1
    while k >= 0 and word[k] == SKIP:
        k -= 1
    return k


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    speical_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # tracemalloc.start()
    data = ""
    with open(input_path, "r") as f:
        data = f.read()

    outputs = re.split("|".join(re.escape(token) for token in speical_tokens),data)
    del data


    sorted = SortedList()
    map1 = dict()
    map2 = dict()
    start1 = time.time()
    print("output length =>", len(outputs))
    for i, output in enumerate(outputs):
        if i%50000 == 0:
            print("we at =>", i, "in time", time.time() - start1)
        # output = "hello world!! this is a beautiful morning"
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, output)
        # print(pre_tokens)
        # pre_tokens =[hello, world, !, !, this, is, a, beatuiful, morning ]
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
    del outputs

    #opt-1
    # In map to, either just store sum of indexes or just store i, iterate on j and k. This way we also reduce memory required by offsets slice. 
    # print(outputs[0])
    # print(outputs_modified)
    # toDelete = list()
    for k , v in map1.items():
        sorted.add(Item(v, k))
    end1 = time.time()
    print(end1 - start1)
    merges = list()
    count = 0
    merge_start = time.time()
    while 256 + len(merges) + len(speical_tokens) < vocab_size:
        # print("done")
        count+=1
        item = sorted[0]
        # if val == 0:
        #     break
        # print(item)
        # cty = map1[(" ",  "a")]
        # idx = sorted.index(Item(cty, (" ", "a")))
        val = item.num
        toLookup  = pair = item.pair
        # print(cty, sorted[idx], val)
        # print("=============================================================")
        # print("max_cnt_token =>", toLookup, "value =>", val)
        token1 = pair[0]
        token2 = pair[1]          
        debug = False
        # if val == 2940:
        #    debug = True  
        #    print("yo yo yo yo", val)
        merges.append(toLookup)
        # print("====>", item, cty)
        # if len(merges) > 10:
        #     break
        # if 256 + len(merges) + len(speical_tokens) > vocab_size:
        #     break
        # key -> [value before first change in this iteration, current value]
        if count%100 == 0:
            print("merge done")
        changed = dict()
        # old map2 key -> new map2 key, for pre-tokens whose segmentation changed in this merge
        changed2 = dict()
        merged = token1 + token2
        for key, cnt in map2.items():
            # keys start as raw bytes (iterating bytes yields ints), merged keys are tuples of tokens
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



    # print(sorted)
    # print(map2)
    # print(map2)                




# profiler = cProfile.Profile()
# profiler.enable()

# # start = time.time()
# [(b' ', b't'), (b' ', b'a'), (b'h', b'e'), (b'i', b'n'), (b' t', b'he'), (b'r', b'e'), (b' ', b'o'), (b' ', b','), (b'e', b'r'), (b' ', b's')]
train_bpe("data/TinyStoriesV2-GPT4-train.txt",10000,  ["<|endoftext|>"])

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("cumulative")
# stats.print_stats(20)

# end = time.time()
# print("DONE")
# print(end - start)
# print(b[:10])
