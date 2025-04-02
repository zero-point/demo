import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Modern content dataset with contributors
modern_content = pd.DataFrame({
    'title': [
        'Ian McKellen', 'Russian Politics', 'Desert Island Discs', 'AI in Theatre', 'Cold War History',
        'Women in Politics', 'Shakespeare Adaptations', 'Classic Radio Dramas', 'WWII Documentaries', 'British Literature'
    ],
    'description': [
        'A discussion on the life and career of Ian McKellen.', 
        'Analysis of Russia’s political landscape today.', 
        'A deep dive into the famous radio show Desert Island Discs.', 
        'Exploring AI’s impact on modern theatre productions.', 
        'A documentary on Cold War conflicts.', 
        'The role of women in shaping modern politics.', 
        'New interpretations of Shakespeare’s works.', 
        'A look back at iconic radio dramas from the 20th century.', 
        'Rare footage and expert analysis of World War II.', 
        'The evolution of British literature and its impact.'
    ],
    'contributor': [
        'Ian McKellen', 'BBC News Analysts', 'Lauren Laverne', 'AI Experts, Theatre Directors', 'Cold War Historians',
        'Politicians, BBC Journalists', 'Royal Shakespeare Company', 'Classic BBC Radio Hosts', 'WWII Historians', 'Literary Scholars'
    ],
    'image': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/SDCC13_-_Ian_McKellen.jpg/1200px-SDCC13_-_Ian_McKellen.jpg',
        'https://www.aljazeera.com/wp-content/uploads/2024/05/2024-05-07T090647Z_692266115_RC2LL7AJBGKF_RTRMADP_3_RUSSIA-POLITICS-PUTIN-1715074442.jpg?resize=1800%2C1800',
        'https://ichef.bbci.co.uk/images/ic/1200x675/p0f7zpfz.jpg',
        'https://easy-peasy.ai/cdn-cgi/image/quality=80,format=auto,width=700/https://media.easy-peasy.ai/6beddadf-a052-4a66-b134-2d38e9251885/9737d998-df81-4176-b3b1-9c657fa273ef.png',
        'https://media.gettyimages.com/photos/berlin-wall-picture-id157387347?s=2048x2048',
        'https://media.gettyimages.com/photos/kamala-harris-speaks-during-a-campaign-event-picture-id1227922017?s=2048x2048',
        'https://media.gettyimages.com/photos/actors-perform-in-a-scene-from-the-play-hamlet-picture-id517331310?s=2048x2048',
        'https://media.gettyimages.com/photos/vintage-radio-picture-id157482029?s=2048x2048',
        'https://media.gettyimages.com/photos/allied-troops-in-landing-craft-picture-id2666288?s=2048x2048',
        'https://media.gettyimages.com/photos/stack-of-old-books-picture-id157482029?s=2048x2048'
    ]
})

# Archive content dataset with contributors
archive_content = pd.DataFrame({
    'title': [
        'THE PROSPECT THEATRE 1969: SHAKESPEARE: THE TRAGEDY OF KING RICHARD II',
        'SOVIET UNION POLITICAL DOCUMENTARY (1978)',
        'BBC RADIO CLASSICS: DESERT ISLAND DISCS 1955',
        'EARLY AI IN MEDIA (1985)',
        'COLD WAR NEWS BROADCAST (1962)',
        'WOMEN IN POLITICS – BBC INTERVIEWS (1970)',
        'SHAKESPEARE ON STAGE – ROYAL THEATRE 1980',
        'GOLDEN AGE OF RADIO (1930s-1950s)',
        'WWII WAR REPORTS – BBC ARCHIVE',
        'BRITISH AUTHORS: A RETROSPECTIVE (1960)'
    ],
    'description': [
        'A 1969 production of Shakespeare’s King Richard II at the Prospect Theatre.',
        'An in-depth political documentary about the Soviet Union in the 1970s.',
        'A classic episode of Desert Island Discs aired in 1955.',
        'A look at how early AI concepts were portrayed in media during the 1980s.',
        'Cold War coverage from the BBC archive, originally broadcast in 1962.',
        'A collection of BBC interviews with female politicians from 1970.',
        'A performance of Shakespearean plays at the Royal Theatre in 1980.',
        'An overview of the golden era of radio broadcasting.',
        'Rare footage and war reports from World War II in the BBC archives.',
        'A documentary covering major British authors and their contributions.'
    ],
    'contributor': [
        'Prospect Theatre Company', 'Soviet Politicians, BBC Reporters', 'Roy Plomley', 'Early AI Scientists',
        'Cold War Journalists', 'Women Politicians from the 1970s', 'Shakespearean Actors', 'BBC Radio Hosts', 'WWII War Reporters', 'British Authors from the 20th Century'
    ],
    'image': [
        'https://mckellen.com/images/1042.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/19191107-lenin_second_anniversary_october_revolution_moscow.jpg/1920px-19191107-lenin_second_anniversary_october_revolution_moscow.jpg',
        'https://ichef.bbci.co.uk/images/ic/320x180/p01hfq78.jpg',
        'https://matthewrenze.com/wp-content/uploads/2020/02/the-history-of-ai.jpg',
        'https://ichef.bbci.co.uk/images/ic/400x225_b/p07td0h4.jpg',
        'https://ichef.bbci.co.uk/ace/standard/976/cpsprodpb/BC08/production/_89463184_hi020216746.jpg',
        'https://cdn2.rsc.org.uk/sitefinity/images/productions/productions-2009-and-before/hamlet/laertes-duels-with-hamlet-1961.tmb-img-820.jpg?sfvrsn=82af3421_1',
        'https://www.swingstreetradio.org/wp-content/uploads/2016/03/oldradio2.jpg',
        'https://media.gettyimages.com/photos/world-war-ii-news-broadcast-picture-id157482029?s=2048x2048',
        'https://media.gettyimages.com/photos/british-author-portrait-picture-id157482029?s=2048x2048'
    ]
})

# Combine title, description, and contributor info
modern_content['text'] = modern_content['title'] + ' ' + modern_content['description'] + ' ' + modern_content['contributor']
archive_content['text'] = archive_content['title'] + ' ' + archive_content['description'] + ' ' + archive_content['contributor']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_modern = vectorizer.fit_transform(modern_content['text'])
tfidf_matrix_archive = vectorizer.transform(archive_content['text'])

# Calculate cosine similarity
similarity_scores = cosine_similarity(tfidf_matrix_modern, tfidf_matrix_archive)
similarity_df = pd.DataFrame(similarity_scores, columns=archive_content['title'], index=modern_content['title'])

# st.title("📻 Flashback")

st.markdown("""
    <div style="display: flex; width: 100%; margin-top: -50px; margin-bottom: -50px; color: white;">
        <div style="margin: 0; width: 70%; text-align: left;"><h1>📻 Flashback</h1></div>
        <div style="margin: 0; width: 30%; text-align: right; padding-left: 5px;"><h3>Your daily dose of personalised archive content</h3></div>
    </div>
""", unsafe_allow_html=True)

selected_show = st.selectbox("Select modern topic:", modern_content['title'])

if selected_show:
    modern_info = modern_content[modern_content['title'] == selected_show].iloc[0]

    # Selected topic section with improved readability
    st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 15px; border-radius: 10px; color: #333;">
            <div style="display: flex; align-items: center;">
                <div style="flex: 1;">
                    <h2 style="color: #000;">{selected_show}</h2>
                    <p><strong>Contributor:</strong> {modern_info['contributor']}</p>
                    <p><strong>Description:</strong> {modern_info['description']}</p>
                </div>
                <div style="flex-shrink: 0; margin-left: 20px;">
                    <img src="{modern_info['image']}" width="120" style="border-radius: 5px;">
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

        # Add a clear separator between sections
    st.divider()

    st.subheader(f"📼 Ranked Recommendations for '{selected_show}':")

    # Rank and show archive recommendations
    recommended_shows = similarity_df.loc[selected_show].sort_values(ascending=False).index[:3]

    # Number emoji for rankings
    ranking_emojis = ['🥇', '🥈', '🥉']

    for rank, show in enumerate(recommended_shows, start=0):
        archive_info = archive_content[archive_content['title'] == show].iloc[0]
        st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 10px 0;">
                <div style="width: 20%; text-align: center;">
                    <h4>{ranking_emojis[rank]}</h4>
                </div>
                <div style="flex: 1;">
                    <h4><strong>{show}</strong></h4>
                    <p><strong>Contributor:</strong> {archive_info['contributor']}</p>
                    <p><strong>Description:</strong> {archive_info['description']}</p>
                </div>
                <div style="flex-shrink: 0; margin-left: 20px;">
                    <img src="{archive_info['image']}" width="200" style="border-radius: 5px;">
                </div>
            </div>
            <hr>
        """, unsafe_allow_html=True)
