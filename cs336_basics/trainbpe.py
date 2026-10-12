import regex as re
from sortedcontainers import SortedList
from functools import total_ordering
import os
import time
import cProfile
import pstats
import pprint
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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    speical_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    data = ""
    with open(input_path, "r") as f:
        data = f.read()

    outputs = re.split("|".join(re.escape(token) for token in speical_tokens),data)

    sorted = SortedList()
    map1 = dict()
    map2 = dict()
    offsets = list()
    outputs_modified = list()
    start1 = time.time()
    print("output length =>", len(outputs))
    for i, output in enumerate(outputs):
        if i%10000 == 0:
            print("we at =>", i, "in time", time.time() - start1)
        # output = "hello world!! this is a beautiful morning"
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, output)
        offset_inner = list()
        ouputs_inner = list()
        # print(pre_tokens)
        # pre_tokens =[hello, world, !, !, this, is, a, beatuiful, morning ]
        for j, pre_token in enumerate(pre_tokens):
            # pre_token= [hello]
            # print(pre_token,  i, j)
            pre_token_bytes = pre_token.encode("utf-8")
            offset_inner_inner = list()
            outputs_inner_inner = list()
            for k in range(len(pre_token_bytes)):
                outputs_inner_inner.append(bytes([pre_token_bytes[k]]))
                offset_inner_inner.append(1)
                if k == len(pre_token_bytes) - 1:
                    break
                toLookup = (bytes([pre_token_bytes[k]]), bytes([pre_token_bytes[k+1]]))
                # if toLookup == target:
                #     print("yo yo yo yo")
                if toLookup in map1:
                    # sorted.discard(Item(val, toLookup))
                    map1[toLookup]+=1
                    # sorted.add(Item(val + 1, toLookup))
                    # if toLookup == (" ", "a"):
                    #     print(val)
            
                    map2[toLookup].append((i, j, k))
                else:
                    # sorted.add(Item(1, toLookup))
                    map2[toLookup] = [(i, j, k)]      
                    map1[toLookup] = 1      
            # offset_inner.append(offset_inner_inner)
            # ouputs_inner.append(outputs_inner_inner)
        offsets.append(offset_inner)
        outputs_modified.append(ouputs_inner)    
    # print(outputs[0])
    # print(outputs_modified)
    # toDelete = list()
    for k , v in map1.items():
        sorted.add(Item(v, k))
    end1 = time.time()
    print(end1 - start1)
    merges = list()
    while 256 + len(merges) + len(speical_tokens) < vocab_size:
        # print("done")
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
        idxList = map2[toLookup].copy()
        merges.append(toLookup)
        # print("====>", item, cty)
        # if len(merges) > 10:
        #     break
        # if 256 + len(merges) + len(speical_tokens) > vocab_size:
        #     break
        oldVals = map1.copy()
        for idx in idxList:
            i, j, k = idx
            # if debug:
                # print("navigating index =>", i,j,k)
                # print("outputs_modfied =>", outputs_modified[i][j])
            if outputs_modified[i][j][k] == "|<end of text>|":
                continue
            # print("navigating index =>", i,j,k)
            # print("outputs_modfied =>", outputs_modified[i][j])
            val = map1[toLookup]
            map1[toLookup]-=1
            if debug:
                print("discarding item", toLookup)
            # sorted.discard(Item(val, toLookup))
            # sorted.add(Item(val-1, toLookup))
            fwdIdx = k + len(token1) + len(token2)
            outputs_modified[i][j][k] = token1 + token2
            outputs_modified[i][j][k + len(token1)] = "|<end of text>|"
            if fwdIdx < len(outputs_modified[i][j]):
                offsets[i][j][fwdIdx] = offsets[i][j][fwdIdx] + offsets[i][j][k + len(token1)] 
                toAdd = (token1+ token2 , outputs_modified[i][j][fwdIdx])
                toRemove = (token2, outputs_modified[i][j][fwdIdx])
                # print("fwdToAdd  =>", toAdd, "fwdToRemove", toRemove)
                # operations for toAdd
                currVal1 = map1.get(toAdd,  0)
                map1[toAdd] =  map1.get(toAdd, 0) + 1
                if currVal1 > 0:
                    map2[toAdd].append((i,j,k))
                    if debug:
                        print("discarding item", toLookup)
                    # sorted.discard(Item(currVal1, toAdd))
                    # sorted.add(Item(currVal1 +1 , toAdd))
                else:
                    map2[toAdd] = [(i,j,k)]
                    # sorted.add(Item(1, toAdd))


                # operations for toRemove
                currVal2 = map1[toRemove]
                map1[toRemove]-=1
                if debug:
                    print("discarding item", toLookup)
                # sorted.discard(Item(currVal2, toRemove))
                # sorted.add(Item(currVal2-1, toRemove))
                map2[toRemove].remove((i, j, k + len(token1)))
            bwdIdx = k -  offsets[i][j][k]
            if bwdIdx >= 0:
                toAdd = (outputs_modified[i][j][bwdIdx], token1 + token2)
                # if toAdd[0] == "?":
                #     print("hereeeee")
                toRemove =   (outputs_modified[i][j][bwdIdx], token1)
                # print("bwdToAdd  =>", toAdd, "bwdToRemove", toRemove)
                currVal1 = map1.get(toAdd, 0)
                map1[toAdd] =  map1.get(toAdd, 0) + 1
                if currVal1 > 0:
                    # print("adding =>", toAdd, i, j, bwdIdx)
                    map2[toAdd].append((i,j,bwdIdx))
                    if debug:
                        print("discarding item", toLookup)
                    # sorted.discard(Item(currVal1, toAdd))
                    # sorted.add(Item(currVal1 +1 , toAdd))
                else:
                    map2[toAdd] = [(i,j,bwdIdx)]
                    # sorted.add(Item(1, toAdd))

                # operations for toRemove
                currVal2 = map1[toRemove]
                map1[toRemove]-=1
                if debug:
                    print("added value => bwd", currVal2 - 1, toRemove)
                if debug:
                    print("discarding item", toLookup)    
                # sorted.discard(Item(currVal2, toRemove))
                # sorted.add(Item(currVal2 -1, toRemove))
                map2[toRemove].remove((i, j, bwdIdx))

        for k ,v in map1.items():
            if k in oldVals:
                old = oldVals[k]
                if old != v:
                    sorted.discard(Item(old, k))
                    sorted.add(Item(v, k))  
            else:
                sorted.add(Item(v,k))           
        del map1[toLookup]
        del map2[toLookup]
        sorted.discard(Item(0, pair))

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

# better allocation
# we at => 0 in time 0.00026607513427734375
# we at => 10000 in time 5.34400200843811
# we at => 20000 in time 9.590490818023682
# we at => 30000 in time 14.271063089370728
# we at => 40000 in time 19.461002111434937
# we at => 50000 in time 24.05393385887146
# we at => 60000 in time 29.185837745666504
# we at => 70000 in time 34.68720102310181
# we at => 80000 in time 38.09470796585083
# we at => 90000 in time 46.7525110244751
# we at => 100000 in time 49.97850298881531

# poor allocation
# we at => 0 in time 0.00010991096496582031
# we at => 10000 in time 5.262231111526489
# we at => 20000 in time 9.56543493270874
# we at => 30000 in time 13.896559953689575
# we at => 40000 in time 17.949470043182373
# we at => 50000 in time 22.264418125152588
# we at => 60000 in time 26.9840829372406
# we at => 70000 in time 32.76332902908325
# we at => 80000 in time 36.28113007545471
# we at => 90000 in time 42.9139769077301
# we at => 100000 in time 46.46346688270569

# with map lookup included


# we at => 0 in time 0.00013589859008789062
# we at => 10000 in time 5.2997517585754395
# we at => 20000 in time 10.268961906433105
# we at => 30000 in time 15.406102895736694
# we at => 40000 in time 20.220227003097534
# we at => 50000 in time 25.301964044570923
# we at => 60000 in time 30.87917995452881
# we at => 70000 in time 37.65780806541443
# we at => 80000 in time 42.08470606803894
# we at => 90000 in time 49.30053496360779
# we at => 100000 in time 53.65579700469971