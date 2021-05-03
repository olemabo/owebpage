from django.test import TestCase

# Create your tests here.
import pandas as pd

class fdr_eliteserien:
    """
        Class of a team
    """

    def __init__(self, gws, oppTeamHomeAwayList, oppTeamNameList, oppTeamDifficultyScore):
        """
            Initialize object
        """
        self.gw = gws
        self.oppTeamHomeAwayList = oppTeamHomeAwayList
        self.oppTeamNameList = oppTeamNameList
        self.oppTeamDifficultyScore = oppTeamDifficultyScore

def create_data_frame_Eliteserien():
    df = pd.read_excel(r'JSON_DATA/Eliteserien_fixtures.xlsx', engine='openpyxl')
    num_rows = len(df.index)
    num_cols = len(df.columns)
    data = []
    for col in range(1, num_cols):
        gws, oppTeamHomeAwayList, oppTeamNameList, oppTeamDifficultyScore = [], [], [], []
        for row in range(num_rows):
            value = df[col][row].split(",")
            df[col][row] = [value[0], value[1], int(value[2])]
            gws.append(col)
            oppTeamHomeAwayList.append(value[1])
            oppTeamNameList.append(value[0])
            oppTeamDifficultyScore.append(value[2])
        data.append(fdr_eliteserien(gws, oppTeamHomeAwayList, oppTeamNameList,oppTeamDifficultyScore))

    return df, data

df = create_data_frame_Eliteserien()
print(df)