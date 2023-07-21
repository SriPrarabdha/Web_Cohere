import cohere
from data_extractor import loader

co = cohere.Client('xCRrVzZjuPM5HN6WFKM1eykBBwHMezrMhaQ0AaD7')

data = loader.load()

dataset= []
for i in range(len(data)):
    dataset.append({
        "prompt" : data[i].metadata["title"] + data[0].metadata["description"],
        "completion" : data[i].page_content
    })
