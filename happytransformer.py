from happytranformer import HappyROBERTA, HappyXLNET

model = HappyXLNET('xlnet-base-cased')
#happy_roberta = HappyROBERTA("roberta-large") # roberta-base/ roberta-large

s = "The fireman is rescuing the [MASK]"
results = happy_roberta.predict_mask(s, num_results=1000)
print(results)
for k,item in enumerate(results):
  if results[k]["word"] == "grandmother":
    print(k, results[k]["softmax"])
    
    
text = "Can you please pass the [MASK] "
options = ["pizza", "rice", "tofu", 'eggs', 'milk']
results = happy_roberta.predict_mask(text, options=options, num_results=3)

print(results)
