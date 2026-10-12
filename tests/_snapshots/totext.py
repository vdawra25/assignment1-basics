# import pickle
# import pprint

# with open("tests/_snapshots/test_train_bpe_special_tokens.pkl", "rb") as f:
#     data = pickle.load(f)

# with open("tests/_snapshots/test_train_bpe_special_tokens.txt", "w") as f:
#     pprint.pprint(data, stream=f)


op1 = dict()
for i in range(256):
    op1[i] = bytes([i])  

print(op1.keys())
print(op1.values())