import sys
import json
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

def getBearerToken():
    res = ""
    with open("keys/Bearer_token") as f:
        res += f.readline()
    return res

def bearer_oauth(r):
    r.headers["Authorization"]=f"Bearer {getBearerToken()}"
    r.headers["User-Agent"]= "v2FilteredStreamPython"
    return r

def cleanTweet(tweet: str) -> str:
    tweet = re.sub(r'http\S+', '', str(tweet))
    tweet = re.sub(r'bit.ly/\S+', '', str(tweet))
    tweet = tweet.strip('[link]')

    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))

    tweet = re.sub(' +', ' ', str(tweet))
    tweet = re.sub('\\n', '', str(tweet))
    tweet = re.sub('#', '', str(tweet))
    tweet = re.sub(':', '', str(tweet))
    tweet = tweet.strip()
    return tweet

def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))

def set_rules(hash_tags):
    sample_rules = [
        {"value" : hash_tags+" lang:en"},
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))

def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 3))

    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(int)])
    

    txts = []
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def cluster(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    initial_batch = vectorizer.fit_transform(data)
    print(initial_batch.shape)
    km = KMeans(n_clusters = 3, init="k-means++", max_iter=100, n_init=1)
    labels = km.fit_predict(initial_batch)
    transformed_data = TSNE(random_state=RC).fit_transform(initial_batch)
    sns.palplot(np.array(sns.color_palette("hls", 3)))
    scatter(transformed_data, labels)
    file_name = "./tmp/initial.png"
    plt.savefig(file_name, dpi=120)

def get_stream():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    
    
    data = []
    count = 0
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            tweet = cleanTweet(json_response['data']['text'])
            count += 1
            if count > 1000:
                break
            if tweet:
                data.append(tweet)
    
    return data
                

if __name__ == "__main__":
    length = len(sys.argv)

    if (length != 2):
        print("Please check following Usage:")
        print("Usage: <script_name> <comma_separated_hash_tags>")
        print()
        exit(1)
    
    hashTags = sys.argv[1].split(',')
    rules = get_rules()
    deleted_rules = delete_all_rules(rules)
    s_rules = set_rules(" OR ".join(hashTags))
    data = get_stream()
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth":2.5})
    RC = 25111993
    cluster(data)

    
    


