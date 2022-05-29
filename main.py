import streamlit as st
import pickle
import pandas as pd

movie_dict = pickle.load(open('movie_list.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)
cosine_sim = pickle.load(open('similarity_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))


def recommended_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    movies_list = movies['title_x'].iloc[movie_indices]
    for i in movies_list:
        st.write(i)


st.title('Movie Recommendation System')

st.text('')

st.header('Motivation and overview')

st.text('')

st.markdown("""The need for recommendation engines is of paramount importance, specifically in the case of
apps/websites with content delivering models. Apart from that, the idea of consuming media pertaining to suggestions is 
very intuitive. Hence, the idea to develop such a system is trivial and has huge applications. Our primary aim behind
this project is to recommend movies based upon a selection that the user provides. This shall serve as a base 
understanding of how technological behemoths such as Netflix and Spotify develop their recommendations engine. We shall 
try to achieve this by finding similarities between movies using mathematical modelling of their characteristics. 
We will use python libraries and packages such as 'pandas','numpy' and 'scikit learn' for ease of implementation of our 
ideas. Throughout the project we shall also analyze the algorithms that are working in the background.""")

st.text('')

st.header('Dataset')

st.text('')

st.markdown("""Before we start with our modelling we need to select a dataset to work on. We need a dataset big enough
to validate our model, but not big enough to make our processing and memory requirements tedious. Keeping these 
constraints in mind, we choose 'tmdb's 5000 movies' dataset. In this dataset we have two csv files. We convert these
 files to dataframes for ease of use.""")

st.text('')

st.text('Code to convert csv to pandas dataframes : ')

code = '''data_frame1 = pd.read_csv('tmdb_5000_movies.csv')
data_frame2 = pd.read_csv('tmdb_5000_credits.csv')'''
st.code(code, language='python')

st.header('Data-filtering and coagulating')

st.text('')

st.markdown("""To process our data with ease we also merge our two dataframes into one with the help of the id 
attribute. The information given to us in the dataset is very extensive. Not all of this information is useful to 
us in formation of our model. For eg: The budget of a movie has little to no implication on whether a person likes it.
Therefore, it can't be a parameter in our model. We hence, narrow our parameters to 'cast', 'overview', 'keywords',
'genre' and 'crew'.""")

st.text('')

st.markdown("""However in case of 'cast' and 'crew' we cannot take everything into account because the extras in cast
and other crew members like make-up artists have little contribution to how the film is received. So we will only
consider the director in 'crew' and the 5 lead actors in case of 'cast'.""")

st.text('')

st.markdown("""To make our data ready for modelling we remove all commas, join space-separated names(so they aren't
mis-understood as two different words), make all words lower-case and separate all words into individual words, rather
than as a string. Now this is not an exhaustive list and we shall skip some of the intricacies but broadly we make the
aforementioned changes. Finally we conjoin all our data and introduce a new column 'tags'. Some snippets of the code 
we shall use are shown below.""")

st.text('')

st.text('Function to get the 5 lead actors in place of cast : ')

code = '''def getactors(x):
    if(isinstance(x,list)):
        actorname = [obj['name'] for obj in x]

        if(len(actorname) > 5):
            return actorname[0:5]
        else:
            return actorname
    
    return []'''
st.code(code, language='python')

st.text('')

st.text('Function to get the name of director : ')

code = '''def getdirector(x):
    if(isinstance(x,list)):
        for obj in x:
            if(obj['job'] == 'Director'):
                return obj['name']
        return np.nan'''
st.code(code, language='python')

st.text('')

st.text('Function to remove commas and make words lowercase')

code = '''def filteringdata(x):
    if(isinstance(x,list)):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if(isinstance(x,str)):
            return str.lower(x.replace(" ",""))
        else:
            return ""'''
st.code(code, language='python')

st.text('')

st.text('Code to separate string into words')

code = '''data_frame1['overview'] = data_frame1['overview'].apply(lambda x:x.split())'''
st.code(code, language='python')

st.text('')

st.text('Code to generate a column called "tags" to hold the data')

code = '''def tag_generator(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['Director'] + ' ' + ' '.join(x['genres']) 
    + ' ' + ' '.join(x['overview'])
data_frame1['tags'] = data_frame1.apply(tag_generator, axis=1)'''
st.code(code, language='python')

st.header('Vectorization of data')

st.text('')

st.markdown("""Now, we have our required data in the 'tags' column of our dataframe which we shall use to compare the 
movies with each other. The challenge we now tackle is of finding an optimal way to compare movies which mediates the 
trade-off of time vs memory.""")

st.markdown("""In algorithmic terms, a brute force approach solution to our problem would be to compare each word of the
selected movie(processed at runtime according to user mandated input) with each word of every other movie and keep
tally. We would then enumerate movies in the decreasing order of that tally.""")

st.markdown("""This however, would be a very tedious affair in terms of computational efficiency, leading to measurable 
delay between the user giving input and the output being displayed on screen. In mathematical terms : """)

st.latex(r'''
Let\hspace{0.2cm}the\hspace{0.2cm}average\hspace{0.2cm} length\hspace{0.2cm} of\hspace{0.2cm} string\hspace{0.2cm} 
be\hspace{0.2cm} S.
\\Let \hspace{0.2cm}the \hspace{0.2cm}number \hspace{0.2cm}of \hspace{0.2cm}words \hspace{0.2cm}in 
\hspace{0.2cm}all 
\hspace{0.2cm}data \hspace{0.2cm}be \hspace{0.2cm}N.
\\Then, \hspace{0.2cm}the \hspace{0.2cm}time \hspace{0.2cm}complexity \hspace{0.2cm}of \hspace{0.2cm}the 
\hspace{0.2cm}computation \hspace{0.2cm}at \hspace{0.2cm}each \hspace{0.2cm}search \hspace{0.2cm}will 
\hspace{0.2cm}be \hspace{0.2cm}in
\hspace{0.2cm}the \hspace{0.2cm}order \hspace{0.2cm}O(NS)''')

st.markdown("""We can tackle the problem of output lag at runtime by comparing the movies beforehand, but that 
pre-computation would also be tedious. Our efforts are also wasted as we our processing a single word multiple times.
Rather than doing this, we can first find out the number of unique words in for dataset and then assign a vector to each 
movie based upon the recurrences of those words in it's 'tags' column. For eg. if a movie 'X' has a word 'Y', 'Z' times
in it's description, we assign it a value of 'Z' in that particular dimension. We can further optimize this by reducing 
the number of unique words. For eg. 'Dances' and 'Dancing' imply the same thing, so we can reduce them to a single word.
After this vector representation is complete, we then calculate the cosine of the angle between these said vectors to 
determine how alike a pair of movies are. A cosine value of 1 implies that the movies are exactly same, whereas a 
cosine value of 0 implies that they are very dissimilar. Although in theory the minimum of cosine value is -1, we won't 
encounter that in our data, as we don't have any negative values.""")

st.markdown("""We calculate the cosine values of all the movies pairwise and thus obtain a 2-D cosine matrix. Now all 
that is left is to sort that matrix movie wise and display the top 10 movies.""")

st.markdown("""The time complexity of this approach is very efficient. In mathematical terms :""")

st.latex(r'''
Let\hspace{0.2cm} the\hspace{0.2cm} average\hspace{0.2cm} length\hspace{0.2cm} of\hspace{0.2cm} a\hspace{0.2cm} 
string\hspace{0.2cm} be\hspace{0.2cm} S.
\\Let\hspace{0.2cm} the\hspace{0.2cm} number\hspace{0.2cm} of\hspace{0.2cm} unique\hspace{0.2cm} words\hspace{0.2cm} 
be\hspace{0.2cm} N.
\\Then\hspace{0.2cm} the\hspace{0.2cm} total\hspace{0.2cm} time\hspace{0.2cm} complexity(pre-computed)\hspace{0.2cm} 
is\hspace{0.2cm} O(NS)''')

st.markdown(""""Now it may seem like both the TCs are the same, but the remember that we would need to compute this 
at every search in the previous case, whereas now we need to compute it only once and query that data everytime 
thereafter.""")

st.text('')

st.text('Importing sckit-learn tools, merging all words into their root words and vectorising the movies')

code = '''from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data_frame1['tags'])'''
st.code(code, language='python')

st.text('')

st.text('Constructing the 2-D cosine matrix')

code = '''from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)'''
st.code(code, language='python')

st.text('')

st.text('Forming indexes to access movies on query and the final recommendation function')

code = '''indices = pd.Series(data_frame1.index, index=data_frame1['title_x']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data_frame1['title_x'].iloc[movie_indices]
'''
st.code(code, language='python')

st.text('')

st.header('Recommendation Engine')

choice = st.selectbox('Select a movie', movies['title_x'].values)
if st.button('Show Recommendation'):
    value = choice
    recommended_movies(value)
