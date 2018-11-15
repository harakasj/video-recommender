import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words('english'))

clips = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-clips.csv')
clip_categories = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-clip-categories.csv')
categories = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-categories.csv')

clip_categories['cat'] = [[int(y) for y in x.split(', ')] for x in clip_categories['categories']]
category_map = {x: y for x,y in zip(categories['category_id'],categories['name'])}
category_map_inv = {y: x for x,y in zip(categories['category_id'],categories['name'])}


# Helper functions

# Map a category name to a category id
def get_cat_name(n, category_map=category_map):
    return ' '.join([category_map[m].lower().replace(' ', '') for m in n])

# Map a category id to a category name
def get_cat_id(n, category_map=category_map):
    return [category_map_inv[m] for m in n]


def get_clip(cid, dataframe):
    return dataframe.loc[cid]

# Remove stop-words from text
def remove_stop_words(text):
    return ' '.join([w for w in word_tokenize(text) if not w in stop_words])


def preprocess(df):
    df.drop(columns=
            ['Unnamed: 0', 'id', 'created', 'filesize', 'duration',
             'total_comments', 'total_plays', 'total_likes', 'thumbnail'],
            inplace=True)

    df.set_index('clip_id', inplace=True)

    clip_cat = clip_categories.copy()
    clip_cat.set_index('clip_id', inplace=True)
    df['cat'] = clip_cat['cat_name']

    # Make everything lowercase
    df['title'] = df['title'].str.lower()
    df['caption'] = df['caption'].str.lower()
    df['title'] = df['title'].str.lower()

    # Remove control characters
    df = df.replace(r'\n|\r', ' ', regex=True)
    df = df.replace(r'[^\w\.\w|\s|]|(\.$)', ' ', regex=True)

    # Fill NaNs with empty string
    df.fillna('', inplace=True)

    columns = ['title', 'cat']
    df['words'] = ""
    for idx, row in df.iterrows():
        w = ''
        for col in columns:
            w = w + row[col] + ' '
        row['words'] = w

    return df


def get_word_vector(df):
    # Corpus is coming from the "words" of each clip - { title, categories, caption }
    corpus = df['words'].tolist()

    cv = CountVectorizer(max_df=0.95)
    word_vector = cv.fit_transform(corpus)

    # Create a TF-IDF document matrix
    tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf.fit(word_vector)

    feature_names = cv.get_feature_names()

    # Generate a list of keywords from the corpus for each clip
    df['keywords'] = ''
    for idx, row in df.iterrows():
        vector = tfidf.transform(cv.transform([row['words']]))
        tfidf_coo = vector.tocoo()
        tfidf_sorted = sorted(zip(tfidf_coo.col, tfidf_coo.data),
                              key=lambda x: (x[1], x[0]), reverse=True)
        keywords = ''
        for idx, score in tfidf_sorted[:10]:
            keywords = keywords + ' ' + feature_names[idx]
        row['keywords'] = keywords

    return df, word_vector


def recommend(clip_id, df, matrix):
    indices = pd.Series(df.index)
    recommended = []
    idx = indices[indices == clip_id].index[0]
    score_series = pd.Series(matrix[idx]).sort_values(ascending=False)
    top_10 = score_series.iloc[1:11]

    for i in top_10.index:
        recommended.append(list(df.index)[i])
    print(top_10)
    return recommended, top_10

def print_results(clip_id):
    print(clips.loc[clips])

def print_info(cid):
    vid = df.loc[cid]
    print(vid['title'] + '\n' + vid['caption'])


# Output file to results.json
def run_test(matrix):
    test_keys = [14434107, 249393804, 71964690, 78106175,
                 228236677, 11374425, 93951774, 35616659,
                 112360862, 116368488]

    rec = pd.DataFrame()
    # label = [test_keys[0]] * 1
    for k in test_keys:
        recs, _ = recommend(k, matrix)
        for r in recs:
            #         print(k)
            l = df.loc[r]
            l['recommend'] = k
            #         l.set_index(k)
            rec = rec.append(l)

    rec['recommend'] = rec['recommend'].astype(int)
    results = rec.set_index([rec['recommend'], rec.index]).to_json('results.json', orient='index')




cid=14434107

clip_categories['cat_name'] = clip_categories['cat'].map(lambda x: get_cat_name(x))
# get_cat_name(clip_categories['cat'].iloc[0])
df = clips.copy()
df = preprocess(df)
for idx, row in df.iterrows():
    row['words'] = remove_stop_words(row['words'])

df, word_vector = get_word_vector(df)

M = cosine_similarity(word_vector, word_vector)

recs, _ = recommend(cid, df, M)
print(recs)
rec = get_clip(cid, df)
for r in recs:
    rec = rec.append(get_clip(r, df),ignore_index=True)
print_info(cid)
print_info(recs[0])


# def main():

# if __name__ == '__main__':
    # main()



