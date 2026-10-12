import regex as re
from collections import defaultdict
def main(): 
    tokens = bpe_tokenize(""" low low low low low
lower lower widest widest widest
newest newest newest newest newest newest """)
    print(tokens)
    print(tokenizepreToken(tokens, "low"))

def tokenizepreToken(tokens, word):
    x = list()
    for w in word:
        x.append(w)
    for token in tokens:
        done = False
        # print(x)
        # print(token)
        y = list()
        for i in range(len(x)):
            if (i!= len(x) -1 ) and (x[i] + x[i+1] == token[0] + token[1]):
                # print("yes", x[i] + x[i+1])
                y.append(x[i] + x[i+1])
                done = True
                continue
            elif done == False:
                y.append(x[i])
            done = False
        x = y
    return x            



def get_lex_max(token1, token2):
    str1 = token1[0] + token1[1]
    str2 = token2[0] + token2[1]
    if str1 > str2:
        return token1
    return token2        

def update_dict(mp, seq, token):
    token_str = token[0] + token[1]
    debug = False
    # if token[0] == "o" and token[1] == "w":
    #     debug = True
    for x in seq:
        indices = [match.start() for match in re.finditer(token_str, x)]
        if debug:
            print(x, token_str)
        for idx in indices:
            if debug:
                print(idx)
            mp[token]-=1
            # print(token, x, token_str)
            if idx != 0:
                if debug:
                    print("behin token", (x[idx-1], token_str))
                mp[(x[idx-1], token_str)]+=1
                mp[(x[idx-1], x[idx])]-=1
            if idx + len(token_str) < len(x):
                if debug:
                    print("in front of token", (x[idx + len(token_str) -1], x[idx + len(token_str)]))
                mp[(token_str, x[idx + len(token_str)])]+=1
                mp[(x[idx + len(token_str) -1], x[idx + len(token_str)])]-=1
    l = list()
    for k, v in mp.items():
        if v <= 0:
            l.append(k)
    for item in l:
        mp.pop(item)    
    return mp        

def bpe_tokenize(input: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    x = re.findall(PAT, input)
    # print(x)
    tokens = list()
    mp = defaultdict(int)
    for y in x:
        for i in range(len(y)-1):
            mp[(y[i], y[i+1])]+=1
    for _ in range(6):
        # print(mp)
        mx = -1
        token = tuple()
        for k, v in mp.items():
            if v > mx:
                mx = v
                token = k
            elif v  == mx:
                token = get_lex_max(token, k)

        # print(token)
        tokens.append(token)
        mp = update_dict(mp,x, token)         
    return tokens


if __name__ == "__main__":
    main()