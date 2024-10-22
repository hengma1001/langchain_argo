import sys

sys.path.append("../")

from langchain_argo import ChatArgo

model = ChatArgo()

result = model._generate("What is the largest planet in the solar system? ")
print(type(result), result)
