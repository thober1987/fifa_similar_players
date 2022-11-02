import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import jupyter_black
import streamlit as st

st.set_page_config(page_title="Fifa Nearest Neighbors", layout="wide")

@st.cache
def prepare_data():
    # Run Black for Better Formatting
    jupyter_black.load()
    # Display all columns
    pd.set_option("display.max_columns", None)
    # Importing File
    fifa = fifa = (
    pd.read_csv("players_fifa23.csv").drop_duplicates(subset=["FullName", "Club"]).reset_index())
    # Data Cleaning
    # Setting Full Name as Index
    fifa2 = fifa.set_index("FullName")
    # Selection of numeric attributes that are useful for player recommendation
    fifa3 = fifa2._get_numeric_data()
    fifa4 = fifa3.drop(
        [
            "ID",
            "Growth",
            "TotalStats",
            "BaseStats",
            "ValueEUR",
            "WageEUR",
            "ReleaseClause",
            "ContractUntil",
            "ClubNumber",
            "ClubJoined",
            "OnLoad",
            "NationalNumber",
            "IntReputation",
            "GKDiving",
            "GKHandling",
            "GKKicking",
            "GKPositioning",
            "GKReflexes",
            "STRating",
            "LWRating",
            "LFRating",
            "CFRating",
            "RFRating",
            "RWRating",
            "CAMRating",
            "LMRating",
            "CMRating",
            "RMRating",
            "LWBRating",
            "CDMRating",
            "RWBRating",
            "LBRating",
            "CBRating",
            "RBRating",
            "GKRating",
        ],
        axis=1,
    )
    # Minmax Scaling
    scaler = MinMaxScaler()

    fifa5 = pd.DataFrame(
        scaler.fit_transform(fifa4), columns=fifa4.columns, index=fifa4.index
    )
    # Weighting: Overall-Skill & Potential should get a higher weight
    fifa5["Overall"] = fifa5["Overall"] * 4
    fifa5["Potential"] = fifa5["Potential"] * 2
    # Recommendation System
    # Instanciating of KNN and Fitting with Dataset
    knn = NearestNeighbors(metric="cosine")
    knn.fit(fifa5)

    # Calculation of Distance and Rank with KNN
    distance, rank = knn.kneighbors(fifa5, 11)
    # Buiding a Dict with Player Index and Player Name
    id_name = fifa["FullName"].to_dict()
    # Column with Player Name and Club
    id_name_club = fifa[["FullName", "Club"]].drop_duplicates()
    id_name_club["Name and Club"] = (
        id_name_club.FullName + " " + "(" + id_name_club.Club + ")"
    )
    id_name_club = id_name_club["Name and Club"].to_dict()
    # Building a DataFrame with the Distances between Players
    dist_df = pd.DataFrame(
        columns=[f"rank_{i}" for i in range(1, 11)],
        index=fifa5.index,
        data=distance[:, 1:],
    )
    # Building a DataFrame with Indices of similar Players and mapping it to Players id_name
    similar_df = pd.DataFrame(
        columns=[f"rank_{i}_name" for i in range(1, 11)],
        index=fifa5.index,
        data=rank[:, 1:],
    ).apply(lambda x: x.map(id_name))
    # Building a DataFrame with Indices of similar Players and mapping it to Players id_name with Club
    similar_with_club_df = pd.DataFrame(
        columns=[f"rank_{i}_name_club" for i in range(1, 11)],
        index=fifa5.index,
        data=rank[:, 1:],
    ).apply(lambda x: x.map(id_name_club))
    # Joining the different DataFrames together
    df_data = fifa4.join(similar_df).join(dist_df).join(similar_with_club_df)
    # Droping rows with Nans
    df_data = df_data.dropna(how="any")
    return df_data, list(id_name.values()) #, df_data[player_name].tolist()

def similar_player(player_name):
    player_name = df_data[df_data.index == player_name].index[0]

    ## Bar chart
    plt.figure(figsize=(12, 6))

    Xaxis = df_data.loc[player_name].values[-20:-10]
    Yaxis = df_data.loc[player_name].values[-10:]

    fig = sns.barplot(data=df_data, x=Xaxis, y=Yaxis, palette="mako")
    fig.set_title('Players similar to  " ' + str(player_name) + ' "')
    # plt.show()
    st.pyplot(plt)

    ## Table display

    display_df = pd.concat(
        [
            df_data[df_data.index == player_name],
            df_data.loc[df_data.loc[player_name].values[42:52]],
        ]
    ).iloc[:, :42]

    st.dataframe(display_df)

# df_data, player_names=prepare_data()
df_data, id_names=prepare_data()
# st.write(id_names)  #(df_data.columns.values.tolist())

selected_player=st.selectbox('Select Player', id_names)
if selected_player:
  similar_player(selected_player)  #("Mats Hummels")

