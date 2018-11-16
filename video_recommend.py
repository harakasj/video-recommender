import sys
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words('english'))

clips = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-clips.csv')
clip_categories = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-clip-categories.csv')
categories = pd.read_csv('similar-staff-picks/similar-staff-picks-challenge-categories.csv')

clip_categories['category'] = [[int(y) for y in x.split(', ')] for x in clip_categories['categories']]
category_map = {x: y for x,y in zip(categories['category_id'],categories['name'])}
category_map_inv = {y: x for x,y in zip(categories['category_id'],categories['name'])}


# Helper functions

# Map a category name to a category id
def get_cat_name(n, category_map=category_map):
    return ' '.join([category_map[m].lower().replace(' ', '') for m in n])


# Map a category id to a category name
def get_cat_id(n, category_map=category_map):
    return [category_map_inv[m] for m in n]


# Remove stop-words from text
def remove_stop_words(text):
    '''
    Strip stop words from text
    :param text: (string) text for a clip
    :return: (string) text without stop words
    '''
    return ' '.join([w for w in word_tokenize(text) if not w in stop_words])


def preprocess(df):
    '''
    Do some basic data cleansing
    :param df: dataframe containing clips
    :return: (Dataframe) df
    '''
    df.drop(columns=
            ['Unnamed: 0', 'id', 'created', 'filesize', 'duration',
             'total_comments', 'total_plays', 'total_likes', 'thumbnail'],
            inplace=True)

    df.set_index('clip_id', inplace=True)

    # Merging relevant columns of dataframes together into one

    clip_cat = clip_categories.copy()

    # set index of clip_category
    clip_cat.set_index('clip_id', inplace=True)
    df['category'] = clip_cat['category_name']

    # Make everything lowercase
    df['title'] = df['title'].str.lower()
    df['caption'] = df['caption'].str.lower()
    df['title'] = df['title'].str.lower()

    # Remove control characters
    df = df.replace(r'\n|\r', ' ', regex=True)
    df = df.replace(r'[^\w\.\w|\s|]|(\.$)', ' ', regex=True)

    # Fill NaNs with empty string
    df.fillna('', inplace=True)

    columns = ['title', 'category']

    # Join all text fields for each clip in order to extract keywords

    df['words'] = ""
    for idx, row in df.iterrows():
        w = ''
        for col in columns:
            w = w + row[col] + ' '
        row['words'] = w

    return df


def get_word_vector(df):
    '''
    Returns dataframe and word vector
    :param df: (Dataframe) of clips
    :return: (Dataframe) df, (vector) word_vector
    '''
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
        for idx, score in tfidf_sorted[:20]:
            keywords = keywords + ' ' + feature_names[idx]
        row['keywords'] = keywords

    return df, word_vector


def recommend(clip_id, df, matrix):
    '''
    :param clip_id: clip_id to recommend similar clips
    :param df: dataframe containing clips
    :param matrix: the similarity matrix
    :return: (list) clip_id of 10 similar videos
    '''
    indices = pd.Series(df.index)
    recommended = []
    idx = indices[indices == clip_id].index[0]
    score_series = pd.Series(matrix[idx]).sort_values(ascending=False)
    top_10 = score_series.iloc[1:11]

    for i in top_10.index:
        recommended.append(list(df.index)[i])
    return recommended, top_10


# Output file to results.json
def run_test(df, matrix):
    '''
    :param df: (Dataframe) the dataframe
    :param matrix: the word
    :return: (json string) results of top clip recommendations
    '''
    test_keys = [14434107, 249393804, 71964690, 78106175,
                 228236677, 11374425, 93951774, 35616659,
                 112360862, 116368488]

    rec = pd.DataFrame()
    for k in test_keys:
        recs, _ = recommend(k, df, matrix)
        for r in recs:
            l = clips.loc[clips['clip_id'] == r]
            # The "recommend" column is going to be part of the multi-index
            l['recommend'] = k
            rec = rec.append(l)
    rec['recommend'] = rec['recommend'].astype(int)
    # Set multi-index for json output clip_id : recommendation[i]
    results = rec.set_index([rec['recommend'], rec.index]).to_json('results.json', orient='index')
    return results


def main(cid):

    # Map clip category codes to category names
    clip_categories['category_name'] = clip_categories['category'].map(lambda x: get_cat_name(x))

    # Make a copy of clips to mess around with
    df = clips.copy()

    # Do some basic data cleansing
    df = preprocess(df)

    # Strip out stop-words
    for idx, row in df.iterrows():
        row['words'] = remove_stop_words(row['words'])

    # Get word vector
    df, word_vector = get_word_vector(df)

    # Get similarity matrix
    M = cosine_similarity(word_vector, word_vector)

    # Run test, return results.json
    # run_test(df, M)


    # Get recommendations
    recs, _ = recommend(cid, df, M)
    # print(*recs, sep='\n')
    rec = pd.DataFrame()
    for r in recs:
        # l = df.loc[r]
        l = clips.loc[clips['clip_id'] == r]
        rec = rec.append(l)

    # Return a dictionary (json object)
    rec.set_index('clip_id', inplace=True)
    results = [{k: rec.values[i][v] for v, k in enumerate(rec.columns)} for i in range(len(rec))]
    print(results)
    return results


if __name__ == '__main__':
    results = main(int(sys.argv[1]))
