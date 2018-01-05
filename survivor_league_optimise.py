from __future__ import print_function
import pandas as pd
import requests 
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import json 
from lxml import html
import argparse 

def get_fte_data(year):
    """
    Download and parse data from fivethirtyeight.com (fte) 
    Data contains modelled probabilities for football teams to win a particular match
    
    :param year: year of fte data 
    :returns: DataFrame
    - team: (string) team playing in match 
    - opp: (string) match opponent of team
    - p_win: (float) fte modelled probability of team winning match
    - loc: (string) location of match for team ['home', 'away'] 
    """
    fte_url = 'https://projects.fivethirtyeight.com/soccer-predictions/forecasts/{}_premier-league_matches.json'.format(year)
    fte_json = requests.get(fte_url).json()
    
    fte_records = []
    for match in fte_json:
        fte_records.append({
            'team': match['team1_code'],
            'p_win': match['prob1'],
            'opp': match['team2_code'], 
            'loc': 'home'
           })
        fte_records.append({
            'team': match['team2_code'],
            'p_win': match['prob2'],
            'opp': match['team1_code'],
            'loc': 'away'
           })  
    return pd.DataFrame.from_dict(fte_records)
    
    
def fs_session_login(fs_credentials):
    """
    Login to footballsurvivor.co.uk and persist authenticated credentials for use in further requests
    
    :param credentials: dict containing 'user[email]' and 'user[password]' keys
    :returns: requests session object
    """
    session = requests.Session()
    # send inititial request to retrieve authenticity token
    login_url = 'https://footballsurvivor.co.uk/users/sign_in'
    html_tree = html.fromstring(session.get(login_url).content)
    authenticity_token = html_tree.xpath('//input[@name="authenticity_token"]/@value')[0]
    
    # append authenticity token to credentials and login
    fs_credentials['authenticity_token'] = authenticity_token 
    fs_credentials['commit'] = 'Login' 
    response = session.post(login_url, data=fs_credentials)
    
    assert response.url != login_url, 'login unsuccessful, check credentials'
    return session


def get_fs_data(fs_session, league_url):
    """
    Download and parse data from footballsurvivor.co.uk (fs) 
    Data contains 'gameweek' information, where gameweeks are groups of matches containing every team exactly once
    
    :param fs_session: requests session object containing authenticated fs login credentials 
    :param league_url: url for the fs league which we will return gameweek data
    :returns: DataFrame
    - gameweek: (int) gameweek of match
    - team: (string) team playing in match 
    - opp: (string) match opponent of team 
    - picked: (boolean) was this team picked in this gameweek in fs league
    - loc: (string) location of match for team ['home', 'away'] 
    """
    html_bytes = fs_session.get(league_url).content
    html_tree = html.fromstring(html_bytes)
    
    fs_records = [] 
    for gameweek_header in html_tree.xpath('//h2[@id]'):
        gameweek_id = int(gameweek_header.attrib['id'].split('-')[-1]) 
        gameweek_fixtures = gameweek_header.getnext()
        for fixture in gameweek_fixtures.xpath('.//tr'):
            home_team = fixture.xpath('./td[1]/span[1]/text()')[0].strip('\n')
            away_team = fixture.xpath('./td[3]/span[2]/text()')[0].strip('\n')
            fs_records.append({
                'gameweek': gameweek_id, 
                'team': home_team, 
                'opp': away_team, 
                'picked': 'team-picked' in fixture.xpath('./td[1]')[0].attrib['class'], 
                'loc': 'home'
                })
            fs_records.append({
                'gameweek': gameweek_id, 
                'team': away_team, 
                'opp': home_team, 
                'picked': 'team-picked' in fixture.xpath('./td[3]')[0].attrib['class'], 
                'loc': 'away'
                })
    return pd.DataFrame.from_dict(fs_records)
    

def merge_fs_fte_data(df_fs, df_fte):
    """
    Map long fs team names to abbreviated fte team names, and merge dataframes 
    
    :param df_fs: DataFrame with footballsurvivor.co.uk gameweek data
    :param df_fte: DataFrame with fivethirtyeight.com win probability data
    :returns: merged DataFrame 
    """

    fs_to_fte = {
        'Arsenal': 'ARS', 
        'Bournemouth': 'BOU', 
        'Brighton': 'BHA', 
        'Burnley': 'BRN', 
        'Chelsea': 'CHE', 
        'Everton': 'EVE', 
        'Huddersfield': 'HUD',
        'Leicester': 'LEI', 
        'Liverpool': 'LIV', 
        'Man United': 'MAN', 
        'Man City': 'MNC', 
        'Newcastle': 'NEW', 
        'Palace': 'CRY',
        'Southampton': 'SOU', 
        'Spurs': 'TOT', 
        'Stoke City': 'STK', 
        'Swansea': 'SWA', 
        'Watford': 'WAT', 
        'West Brom': 'WBA', 
        'West Ham': 'WHU' 
    }
    # map team names
    df_fs.loc[:, ['team', 'opp']] = df_fs[['team', 'opp']].applymap(fs_to_fte.get)
    return df_fte.merge(df_fs, on=('team', 'opp', 'loc'))
   
   
def filter_team_gameweek(df, previous_picks=None, forecast_length=None, teams=set(), gameweeks=set()):
    """
    Filter specific gameweeks and teams from input dataframe to limit pick options 
    Options are: (can be used individually or in combination)
     - remove all previously picked teams/gameweeks
     - remove all gameweeks which exceed "forecast" length 
     - pass arbitrary set of gameweeks and teams to be removed 
    
    :param df: DataFrame containing 'team' and 'gameweek' (and 'picked') columns
    :param previous_picks: name of column in df indicating previous picks (boolean) e.g. 'picked' 
    :param forecast_length: number of future gameweeks to preserve 
    :param teams: set of teams to exclude from df  
    :param gameweeks: set of gameweeks to exclude from df
    :returns: filtered DataFrame
    """
    # set start point for forecast period 
    forecast_start = df['gameweek'].min() - 1
    
    if previous_picks:
        picked_teams = df.loc[df.picked, 'team']
        picked_gameweeks = df.loc[df.picked, 'gameweek']
        teams.update(picked_teams) 
        gameweeks.update(picked_gameweeks)
        # update start point for forecast period if dropping previous picks 
        forecast_start = picked_gameweeks.max() 
        
    if forecast_length:
        nopick_gameweeks = df.loc[df.gameweek > forecast_start + forecast_length, 'gameweek'] 
        gameweeks.update(nopick_gameweeks)
    
    print("excluding teams:", teams) 
    print("excluding gameweeks:", gameweeks) 
    return df[~(df.team.isin(teams)) & ~(df.gameweek.isin(gameweeks))]  


def get_probability_matrix(df):
    """
    :param df: DataFrame containing 'team', 'gameweek' and 'p_win' columns
    :returns: reshaped DataFrame with gameweeks as rows, teams as columns and values as probabilities
    """
    return df.set_index(['gameweek', 'team'])['p_win'].unstack()
    
    
def optimise_picks(df, value_label):
    """ 
    Select exactly one row for each column of input DataFrame, such that the sum of the values in the row/column intersection are maximized 
    Number of columns of input DataFrame must be greater than or equal to number of rows
    In the case where number of columns is greater, only n columns will have a selected value, were n = number of rows 
    
    :param df: input DataFrame containing values to be maximized
    :param value_label: description of values contained in df e.g. 'win_probability' 
    :returns: DataFrame with one row corresponding to each selected value 
    """
    cost_matrix = df.values * -1 # taking inverse costs as we want to maximise
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    
    d = defaultdict(list)
    for i in range(min(cost_matrix.shape)):
        row_idx = row_ids[i] 
        col_idx = col_ids[i]
        d[df.index.name].append(df.index[row_idx])
        d[df.columns.name].append(df.columns[col_idx])
        d[value_label].append(cost_matrix[row_idx, col_idx] * -1) 
    return pd.DataFrame(d)
    
    
def plot_picks_heatmap(df_prob_matrix, df_picks, plot_size=(None, None)):
    """
    :param df_prob_matrix: DataFrame with rows as gameweeks, columns as teams and values as win probability 
    :param df_picks: DataFrame with team and gameweek columns (one row per pick) 
    :param plot_size: tuple containing plot dimensions
    """
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    sns.set(rc={'figure.figsize': plot_size}) 
    ax = sns.heatmap(df_prob_matrix, cmap=sns.color_palette("Blues", n_colors=20), annot=True, cbar=False)
    for _, row in df_picks.iterrows():
        row_num = df_prob_matrix.index.get_loc(row['gameweek'])
        col_num = df_prob_matrix.columns.get_loc(row['team']) 
        ax.add_patch(Rectangle((col_num, row_num), 1, 1, fill=False, edgecolor='red', lw=2))
    return ax


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('footballsurvivor.co.uk picks optimiser') 
    parser.add_argument('-f', '--forecast', action='store', dest='forecast', default=None, type=int, help='number of future weeks to make picks')
    args = parser.parse_args()
    
    league_url = 'https://footballsurvivor.co.uk/leagues/geo_punters_winner_takes_all/entries/70392/fixtures' 
    with open('fs_credentials.json', 'rb') as cred_file:
        fs_credentials = json.load(cred_file)

    # get fivethirtyeight.com data
    df_fte = get_fte_data(year=2017)
    
    # login and get footballsurvivor.co.uk data
    fs_session = fs_session_login(fs_credentials)
    df_fs = get_fs_data(fs_session, league_url)
    
    # standardise team names and merge dataframes
    df_merged = merge_fs_fte_data(df_fs, df_fte) 
    
    # filter picked teams/gameweeks and set number of future gameweeks to make picks
    df_merged = filter_team_gameweek(df_merged, previous_picks='picked', forecast_length=args.forecast)
    
    # reshape "long" data to "wide" probability matrix 
    df_prob_matrix = get_probability_matrix(df_merged)
    
    # get optimised picks 
    df_picks = optimise_picks(df_prob_matrix, value_label='p_win')
    
    print(df_picks) 
