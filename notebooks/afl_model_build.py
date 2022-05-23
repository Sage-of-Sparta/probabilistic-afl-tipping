
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import feature_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def predict_round(pred_round):

    def load_afl_data(pred_round):
        df_2017 = pd.read_csv("../data/afl_results_2017.csv")
        #print(df_2017.shape)
        df_2018 = pd.read_csv("../data/afl_results_2018.csv")
        #print(df_2018.shape)
        df_2019 = pd.read_csv("../data/afl_results_2019.csv")
        #print(df_2019.shape)
        df_2020 = pd.read_csv("../data/afl_results_2020.csv")
        #print(df_2020.shape)
        df_2021 = pd.read_csv("../data/afl_results_2021.csv")
        #print(df_2021.shape)
        df_2022 = pd.read_csv("../data/afl_results_2022.csv")
        pred_round_results = df_2022[df_2022['round.roundNumber'] == pred_round]
        df_2022 = df_2022[df_2022['round.roundNumber'] < pred_round]
        
        #print(df_2022.shape)
        df_all = pd.concat([df_2017, df_2018, df_2019, df_2020, df_2021,df_2022], axis=0)
        df_all['Date'] = pd.to_datetime(df_all['match.date']).dt.strftime("%Y-%m-%d")
        df_players_2017 = pd.read_csv("../data/afl_players_stats_2017.csv")
        #print(df_players_2017.shape)
        df_players_2018 = pd.read_csv("../data/afl_players_stats_2018.csv")
        #print(df_players_2018.shape)
        df_players_2019 = pd.read_csv("../data/afl_players_stats_2019.csv")
        #print(df_players_2019.shape)
        df_players_2020 = pd.read_csv("../data/afl_players_stats_2020.csv")
        #print(df_players_2020.shape)
        df_players_2021 = pd.read_csv("../data/afl_players_stats_2021.csv")
        #print(df_players_2021.shape)
        df_players_2022 = pd.read_csv("../data/afl_players_stats_2022.csv")
        df_players_2022 = df_players_2022[df_players_2022['Round'] < pred_round]
        
        #print(df_players_2022.shape)
        df_players = pd.concat([df_players_2017, df_players_2018, df_players_2019,df_players_2020,df_players_2021,df_players_2022], axis=0)
        #print(df_players.shape)
        #df_players.columns
        
        df_fixture = pd.read_csv("../data/fixture_2022.csv")
        df_next_games_teams = df_fixture[(df_fixture['round.roundNumber'] == pred_round)]
        df_next_games_teams = df_next_games_teams[['home.team.name','away.team.name','venue.name','compSeason.year','round.roundNumber']]
        df_next_games_teams = df_next_games_teams.rename(columns={'home.team.name': 'match.homeTeam.name', 'away.team.name': 'match.awayTeam.name','compSeason.year':'round.year'})
        df_next_games_teams['match.matchId'] = np.arange(len(df_next_games_teams))
        
        return df_all, df_players, df_fixture, df_next_games_teams, pred_round_results

    def get_aggregate_player_stats(df=None):

        agg_stats = (df.rename(columns={ # Rename columns to lowercase
                        'Home.team': 'match.homeTeam.name',
                        'Away.team': 'match.awayTeam.name',
                        })
                       .groupby(by=['Date', 'Season', 'match.homeTeam.name', 'match.awayTeam.name'], as_index=False) # Groupby to aggregate the stats for each game
                       .sum()
                       #.drop(columns=['DE', 'TOG', 'Match_id']) # Drop columns
                       .assign(date=lambda df: pd.to_datetime(df.Date, format="%Y-%m-%d")) # Create a datetime object
                       .sort_values(by='Date')
                       .reset_index(drop=True))
        return agg_stats


    df_all, df_players, df_fixture, df_next_games_teams, pred_round_results = load_afl_data(pred_round)

    agg_player = get_aggregate_player_stats(df_players)
    afl_df = df_all.merge(agg_player, on=['Date', 'match.homeTeam.name', 'match.awayTeam.name'], how='left')

    # Add average goal diff for home and away team rolling 4 games

    afl_df['HTGDIFF'] = afl_df['homeTeamScore.matchScore.goals'] - afl_df['awayTeamScore.matchScore.goals']
    afl_df['ATGDIFF'] = afl_df['awayTeamScore.matchScore.goals'] - afl_df['homeTeamScore.matchScore.goals']

    def from_dict_value_to_df(d):
        """
        input = dictionary 
        output = dataframe as part of all the values from the dictionary
        """
        df = pd.DataFrame()
        for v in d.values():
            df = pd.concat([df,v])
        return df

    def avg_goal_diff(df, avg_h_a_diff, a_h_team, a_h_goal_letter):
        """
        input: 
            df = dataframe with all results
            avg_h_a_diff = name of the new column
            a_h_team = HomeTeam or AwayTeam
            a_h_goal_letter = 'H' for home or 'A' for away
        output: 
            avg_per_team = dictionary with with team as key and columns as values with new column H/ATGDIFF
        """
        df[avg_h_a_diff] = 0
        avg_per_team = {}
        all_teams = df[a_h_team].unique()
        for t in all_teams:
            df_team = df[df[a_h_team]==t].fillna(0)
            result = df_team['{}TGDIFF'.format(a_h_goal_letter)].rolling(4).mean()
            df_team[avg_h_a_diff] = result
            avg_per_team[t] = df_team
        return avg_per_team

    d_AVGFTHG = avg_goal_diff(afl_df, 'AVGHTGDIFF', 'match.homeTeam.name', 'H')
    df_AVGFTHG = from_dict_value_to_df(d_AVGFTHG)
    df_AVGFTHG.sort_index(inplace=True)
    d_AVGFTAG = avg_goal_diff(df_AVGFTHG, 'AVGATGDIFF', 'match.awayTeam.name', 'A')
    afl_df = from_dict_value_to_df(d_AVGFTAG)
    afl_df.sort_index(inplace=True)
    afl_df['AVGATGDIFF'].fillna(0, inplace=True)

    afl_df['goal_diff'] = afl_df['homeTeamScore.matchScore.goals'] - afl_df['awayTeamScore.matchScore.goals']

    for index, row in df_all[df_all['match.status']=='CONCLUDED'].iterrows():
        if afl_df['goal_diff'][index] > 0:
            afl_df.at[index,'result'] = 1   # 1 is a win
        else:
            afl_df.at[index,'result'] = 0  # 0 is a loss 

    def previous_data(df, h_or_a_team, column, letter, past_n):
        """
        input: 
            df = dataframe with all results
            a_h_team = HomeTeam or AwayTeam
            column = column selected to get previous data from
        output:
            team_with_past_dict = dictionary with team as a key and columns as values with new 
                                  columns with past value
        """
        d = dict()
        team_with_past_dict = dict()
        all_teams = df[h_or_a_team].unique()
        for team in all_teams:
            n_games = len(df[df[h_or_a_team]==team])
            team_with_past_dict[team] = df[df[h_or_a_team]==team]
            for i in range(1, past_n):
                d[i] = team_with_past_dict[team].assign(
                    result=team_with_past_dict[team].groupby(h_or_a_team)[column].shift(i)
                ).fillna({'{}_X'.format(column): 0})
                team_with_past_dict[team]['{}_{}_{}'.format(letter, column, i)] = d[i].result
        return team_with_past_dict

    def previous_data_call(df, side, column, letter, iterations):
        d = previous_data(df, side, column, letter, iterations)
        df_result= from_dict_value_to_df(d)
        df_result.sort_index(inplace=True)
        return df_result

    df_last_home_results = previous_data_call(afl_df, 'match.homeTeam.name', 'result', 'H', 3)
    df_last_away_results = previous_data_call(df_last_home_results, 'match.awayTeam.name', 'result', 'A', 3)
    df_last_last_HTGDIFF_results = previous_data_call(df_last_away_results, 'match.homeTeam.name', 'HTGDIFF', 'H', 3)
    df_last_last_ATGDIFF_results = previous_data_call(df_last_last_HTGDIFF_results, 'match.awayTeam.name', 'ATGDIFF', 'A', 3)
    df_last_AVGFTHG_results = previous_data_call(df_last_last_ATGDIFF_results, 'match.homeTeam.name', 'AVGHTGDIFF', 'H', 2)
    df_last_AVGFTAG_results = previous_data_call(df_last_AVGFTHG_results, 'match.awayTeam.name', 'AVGATGDIFF', 'A', 2)
    afl_df = df_last_AVGFTAG_results.copy()

    all_cols = ['match.matchId','match.date', 'match.status', 'match.venue', 'match.homeTeam.name', 'match.awayTeam.name','venue.name', 'venue.state', 'round.name', 'round.year', 'round.roundNumber', 'status',
    'homeTeamScore.rushedBehinds', 'homeTeamScore.minutesInFront',
           'homeTeamScore.matchScore.totalScore', 'homeTeamScore.matchScore.goals',
           'homeTeamScore.matchScore.behinds',
           'homeTeamScore.matchScore.superGoals', 'awayTeamScore.rushedBehinds',
           'awayTeamScore.minutesInFront', 'awayTeamScore.matchScore.totalScore',
           'awayTeamScore.matchScore.goals', 'awayTeamScore.matchScore.behinds',
           'awayTeamScore.matchScore.superGoals', 'weather.tempInCelsius',
           'homeTeamScoreChart.goals', 'homeTeamScoreChart.leftBehinds',
           'homeTeamScoreChart.rightBehinds', 'homeTeamScoreChart.leftPosters',
           'homeTeamScoreChart.rightPosters', 'homeTeamScoreChart.rushedBehinds',
           'homeTeamScoreChart.touchedBehinds', 'awayTeamScoreChart.goals',
           'awayTeamScoreChart.leftBehinds', 'awayTeamScoreChart.rightBehinds',
           'awayTeamScoreChart.leftPosters', 'awayTeamScoreChart.rightPosters',
           'awayTeamScoreChart.rushedBehinds', 'awayTeamScoreChart.touchedBehinds', 
           'HQ1G', 'HQ1B', 'HQ2G',
           'HQ2B', 'HQ3G', 'HQ3B', 'HQ4G', 'HQ4B', 'Home.score', 'AQ1G', 'AQ1B',
           'AQ2G', 'AQ2B', 'AQ3G', 'AQ3B', 'AQ4G', 'AQ4B', 'Away.score',
           'Kicks', 'Marks', 'Handballs', 'Goals', 'Behinds', 'Hit.Outs',
           'Tackles', 'Rebounds', 'Inside.50s', 'Clearances', 'Clangers',
           'Frees.For', 'Frees.Against', 'Brownlow.Votes', 'Contested.Possessions',
           'Uncontested.Possessions', 'Contested.Marks', 'Marks.Inside.50',
           'One.Percenters', 'Bounces', 'Goal.Assists', 'Time.on.Ground..',
           'Substitute', 'group_id', 'HTGDIFF', 'ATGDIFF', 'AVGHTGDIFF',
           'AVGATGDIFF', 'goal_diff', 'result', 'H_result_1', 'H_result_2',
           'A_result_1', 'A_result_2', 'H_HTGDIFF_1', 'H_HTGDIFF_2', 'A_ATGDIFF_1',
           'A_ATGDIFF_2', 'H_AVGHTGDIFF_1', 'A_AVGATGDIFF_1']

    non_feature_cols = ['match.matchId','match.date', 'match.status', 'match.venue', 'match.homeTeam.name', 'match.awayTeam.name','venue.name', 'venue.state', 'round.name', 'round.year', 'round.roundNumber', 'status','Season']
    feature_cols = [
           'homeTeamScore.rushedBehinds', 'homeTeamScore.minutesInFront',
           'homeTeamScore.matchScore.totalScore', 'homeTeamScore.matchScore.goals',
           'homeTeamScore.matchScore.behinds',
           'homeTeamScore.matchScore.superGoals', 'awayTeamScore.rushedBehinds',
           'awayTeamScore.minutesInFront', 'awayTeamScore.matchScore.totalScore',
           'awayTeamScore.matchScore.goals', 'awayTeamScore.matchScore.behinds',
           'awayTeamScore.matchScore.superGoals', 'weather.tempInCelsius',
           'homeTeamScoreChart.goals', 'homeTeamScoreChart.leftBehinds',
           'homeTeamScoreChart.rightBehinds', 'homeTeamScoreChart.leftPosters',
           'homeTeamScoreChart.rightPosters', 'homeTeamScoreChart.rushedBehinds',
           'homeTeamScoreChart.touchedBehinds', 'awayTeamScoreChart.goals',
           'awayTeamScoreChart.leftBehinds', 'awayTeamScoreChart.rightBehinds',
           'awayTeamScoreChart.leftPosters', 'awayTeamScoreChart.rightPosters',
           'awayTeamScoreChart.rushedBehinds', 'awayTeamScoreChart.touchedBehinds', 
           'HQ1G', 'HQ1B', 'HQ2G',
           'HQ2B', 'HQ3G', 'HQ3B', 'HQ4G', 'HQ4B', 'Home.score', 'AQ1G', 'AQ1B',
           'AQ2G', 'AQ2B', 'AQ3G', 'AQ3B', 'AQ4G', 'AQ4B', 'Away.score',
           'Kicks', 'Marks', 'Handballs', 'Goals', 'Behinds', 'Hit.Outs',
           'Tackles', 'Rebounds', 'Inside.50s', 'Clearances', 'Clangers',
           'Frees.For', 'Frees.Against', 'Brownlow.Votes', 'Contested.Possessions',
           'Uncontested.Possessions', 'Contested.Marks', 'Marks.Inside.50',
           'One.Percenters', 'Bounces', 'Goal.Assists', 'Time.on.Ground..',
           'Substitute', 'group_id', 'HTGDIFF', 'ATGDIFF', 'AVGHTGDIFF',
           'AVGATGDIFF', 'goal_diff', 'result', 'H_result_1', 'H_result_2',
           'A_result_1', 'A_result_2', 'H_HTGDIFF_1', 'H_HTGDIFF_2', 'A_ATGDIFF_1',
           'A_ATGDIFF_2', 'H_AVGHTGDIFF_1', 'A_AVGATGDIFF_1']

    afl_df = afl_df[all_cols] 

    afl_df = afl_df.rename(columns={col: 'f_' + col for col in afl_df if col not in non_feature_cols})



    def create_training_and_test_data(afl_df,df_next_games_teams):
        
        # Define a function which returns a DataFrame with the expontential moving average for each numeric stat
        def create_exp_weighted_avgs(df, span):
            # Create a copy of the df with only the game id and the team - we will add cols to this df
            ema_features = df[['match.matchId', 'match.homeTeam.name']].copy()

            feature_names = [col for col in df.columns if col.startswith('f_')] # Get a list of columns we will iterate over

            for feature_name in feature_names:
                feature_ema = (df.groupby('match.homeTeam.name')[feature_name]
                                 .transform(lambda row: (row.ewm(span=span)
                                                            .mean()
                                                            .shift(1))))
                ema_features[feature_name] = feature_ema

            return ema_features

            # Define a function which finds the elo for each team in each game and returns a dictionary with the game ID as a key and the
        # elos as the key's value, in a list. It also outputs the probabilities and a dictionary of the final elos for each team
        def elo_applier(df, k_factor):
            # Initialise a dictionary with default elos for each team
            elo_dict = {team: 1500 for team in df['match.homeTeam.name'].unique()}
            elos, elo_probs = {}, {}

            # Loop over the rows in the DataFrame
            for index, row in df.iterrows():
                # Get the Game ID
                game_id = row['match.matchId']

                # Get the margin
                margin = row['f_goal_diff']

                # If the game already has the elos for the home and away team in the elos dictionary, go to the next game
                if game_id in elos.keys():
                    continue

                # Get the team and opposition
                home_team = row['match.homeTeam.name']
                away_team = row['match.awayTeam.name']

                # Get the team and opposition elo score
                home_team_elo = elo_dict[home_team]
                away_team_elo = elo_dict[away_team]

                # Calculated the probability of winning for the team and opposition
                prob_win_home = 1 / (1 + 10**((away_team_elo - home_team_elo) / 400))
                prob_win_away = 1 - prob_win_home

                # Add the elos and probabilities our elos dictionary and elo_probs dictionary based on the Game ID
                elos[game_id] = [home_team_elo, away_team_elo]
                elo_probs[game_id] = [prob_win_home, prob_win_away]

                # Calculate the new elos of each team
                if margin > 0: # Home team wins; update both teams' elo
                    new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home)
                    new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away)
                elif margin < 0: # Away team wins; update both teams' elo
                    new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home)
                    new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away)
                elif margin == 0: # Drawn game' update both teams' elo
                    new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home)
                    new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away)

                # Update elos in elo dictionary
                elo_dict[home_team] = new_home_team_elo
                elo_dict[away_team] = new_away_team_elo

            return elos, elo_probs, elo_dict
        
        afl_df['train_data'] = 1
        df_next_games_teams['train_data'] = 0
        
        afl_data = afl_df.append(df_next_games_teams).reset_index(drop=True)
        
        features_rolling_averages = create_exp_weighted_avgs(afl_data, span=10)
        
        features = afl_data[['match.date', 'match.matchId', 'match.homeTeam.name', 'match.awayTeam.name', 'venue.name','round.year','train_data']].copy()
        features = pd.merge(features, features_rolling_averages, on=['match.matchId', 'match.homeTeam.name'])
        
        form_btwn_teams = afl_df[['match.matchId', 'match.homeTeam.name', 'match.awayTeam.name', 'f_goal_diff']].copy()


        elos, elo_probs, elo_dict = elo_applier(afl_data, 30)
        # Add our created features - elo, efficiency etc.
        
        features = (features.assign(f_elo_home=lambda df: df['match.matchId'].map(elos).apply(lambda x: x[0]),
                                                    f_elo_away=lambda df: df['match.matchId'].map(elos).apply(lambda x: x[1]))
                                              .reset_index(drop=True))
        
    #    form_btwn_teams_inv = pd.DataFrame()

    #    for index, row in form_btwn_teams.iterrows():
    #        home = row['match.homeTeam.name']
    #        away = row['match.awayTeam.name']
    #        matchid = row['match.matchId']
    #        margin = row['f_goal_diff']

    #        form_btwn_teams_inv = form_btwn_teams_inv.append({'match.matchId': matchid, 'match.homeTeam.name': away, 'match.awayTeam.name': home, 'f_goal_diff': -1*margin}, ignore_index=True)

    #    form_btwn_teams['f_form_margin_btwn_teams'] = (form_btwn_teams.groupby(['match.homeTeam.name', 'match.awayTeam.name'])['f_goal_diff']
    #                                                              .transform(lambda row: row.rolling(5).mean().shift())
    #                                                              .fillna(0))

    #    form_btwn_teams['f_form_past_5_btwn_teams'] = \
    #    (form_btwn_teams.assign(win=lambda df: df.apply(lambda row: 1 if row.f_goal_diff > 0 else 0, axis='columns'))
    #                  .groupby(['match.homeTeam.name', 'match.awayTeam.name'])['win']
    #                  .transform(lambda row: row.rolling(5).mean().shift() * 5)
    #                  .fillna(0))


        #print(features.shape)
        # Merge to our features df
        #features = pd.merge(features, form_btwn_teams_1.drop(columns=['f_goal_diff']), on=['match.matchId', 'match.homeTeam.name', 'match.awayTeam.name'])
        #print(features.shape)
        
        # Get the result and merge to the feature_df

        match_results = (afl_df.assign(result=lambda df: df.apply(lambda row: 1 if row['f_goal_diff'] > 0 else 0, axis=1)))
        # Merge result column to feature_df
        feature_df = pd.merge(features, match_results[['match.matchId', 'result']], on='match.matchId')

        return feature_df,features_rolling_averages, afl_data, features

    feature_df, features_rolling_averages, afl_data, features = create_training_and_test_data(afl_df,df_next_games_teams)
    feature_columns = [col for col in feature_df if col.startswith('f_')]
    #features['f_elo_home'] = features['f_elo_home']/1000
    #features['f_elo_away'] = features['f_elo_away']/1000

    # Build model from feature_df

    feature_df = feature_df.dropna()

    all_X = feature_df.loc[:, feature_columns]
    all_y = feature_df.loc[:, 'result']

    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.30, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_test[feature_columns] = scaler.transform(X_test[feature_columns])

    # Create a list of standard classifiers
    classifiers = [
        #Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        #Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),
        
        #GLM
        linear_model.LogisticRegressionCV(),
        
        #Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
        
        #SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        
        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        #XGBClassifier()    
    ]

    # Define a functiom which finds the best algorithms for our modelling task
    def find_best_algorithms(classifier_list, X, y):
        # This function is adapted from https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
        # Cross validate model with Kfold stratified cross validation
        kfold = StratifiedKFold(n_splits=5)
        
        # Grab the cross validation scores for each algorithm
        cv_results = [cross_val_score(classifier, X, y, scoring = "neg_log_loss", cv = kfold) for classifier in classifier_list]
        cv_means = [cv_result.mean() * -1 for cv_result in cv_results]
        cv_std = [cv_result.std() for cv_result in cv_results]
        algorithm_names = [alg.__class__.__name__ for alg in classifiers]
        
        # Create a DataFrame of all the CV results
        cv_results = pd.DataFrame({
            "Mean Log Loss": cv_means,
            "Log Loss Std": cv_std,
            "Algorithm": algorithm_names
        })
        
        
        return cv_results.sort_values(by='Mean Log Loss').reset_index(drop=True)

    best_algos = find_best_algorithms(classifiers, X_train, y_train)

    # Define a function which optimises the hyperparameters of our chosen algorithms
    def optimise_hyperparameters(train_x, train_y, algorithms, parameters):
        kfold = StratifiedKFold(n_splits=5)
        best_estimators = []
        
        for alg, params in zip(algorithms, parameters):
            gs = GridSearchCV(alg, param_grid=params, cv=kfold, scoring='neg_log_loss', verbose=1)
            gs.fit(train_x, train_y)
            best_estimators.append(gs.best_estimator_)
        return best_estimators

    # Define our parameters to run a grid search over
    lr_grid = {
        "C": [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
        "solver": ["newton-cg", "lbfgs", "liblinear"]
    }

    # Add our algorithms and parameters to lists to be used in our function
    alg_list = [LogisticRegression(), ensemble.RandomForestClassifier()]
    param_list = [lr_grid]

    # Find the best estimators, then add our other estimators which don't need optimisation
    best_estimators = optimise_hyperparameters(X_train, y_train, alg_list, param_list)

    lr_best_params = best_estimators[0].get_params()

    lr = LogisticRegression(**lr_best_params)
    lr.fit(X_train, y_train)
    final_predictions_lr = lr.predict(X_test)

    accuracy = (final_predictions_lr == y_test).mean() * 100

    next_round_features = features[features['train_data']==0][feature_columns]

    next_round_predictions = lr.predict(next_round_features)
    
    prediction_probs = lr.predict_proba(next_round_features)
    
    df_next_games_teams['pred_home_result'] =  next_round_predictions
    df_next_games_teams['pred_home_prob'] = prediction_probs[:,1].round(3)
    
    df_next_games_teams['enter_tips'] = ''
    for i in range(len(df_next_games_teams)):
        pred_home_result = df_next_games_teams['pred_home_result'].values[i]
        
        if pred_home_result == 1:
            entertips = 'pick %s with p=%s' %(df_next_games_teams['match.homeTeam.name'].values[i],df_next_games_teams['pred_home_prob'].values[i])
            df_next_games_teams['enter_tips'].values[i] = entertips
        else:
            entertips = 'pick %s with p=%s' % (df_next_games_teams['match.awayTeam.name'].values[i],1-df_next_games_teams['pred_home_prob'].values[i])
            df_next_games_teams['enter_tips'].values[i] = entertips
        
    
    
    if len(pred_round_results)==0:
        return accuracy, df_next_games_teams, features, afl_df
    
    
    pred_round_results['result'] = np.where(pred_round_results['homeTeamScore.matchScore.totalScore']>pred_round_results['awayTeamScore.matchScore.totalScore'],1,0)
    
    actual_results = pred_round_results[['match.homeTeam.name','match.awayTeam.name','round.roundNumber','homeTeamScore.matchScore.totalScore','awayTeamScore.matchScore.totalScore','result']]
    
    df_next_games_teams = pd.merge(df_next_games_teams, actual_results, on=['match.homeTeam.name', 'match.awayTeam.name'])

    df_next_games_teams['score_1'] = 0.0
    df_next_games_teams['score_2'] = 0.0
    df_next_games_teams['score_3'] = 0.0

    for i in range(len(df_next_games_teams)):
        
        p = df_next_games_teams['pred_home_prob'].values[i] 
        q = df_next_games_teams['pred_home_prob'].values[i] 
        
        
        if p > 0.68:
            p = 0.68
        elif p < 0.32:
            p = 0.32

        if q > 0.8:
            q = 0.8
        elif q < 0.2:
            q = 0.2
            
            
        
        if df_next_games_teams['homeTeamScore.matchScore.totalScore'].values[i] == df_next_games_teams['awayTeamScore.matchScore.totalScore'].values[i]:
            df_next_games_teams['score_1'].values[i] = 1.0 + 0.5 * np.log2(p*(1-p))
            df_next_games_teams['score_2'].values[i] = 1.0 + 0.5 * np.log2(p*(1-p))
            df_next_games_teams['score_3'].values[i] = 1.0 + 0.5 * np.log2(q*(1-q))
            
        elif (df_next_games_teams['pred_home_result'].values[i] == df_next_games_teams['result'].values[i]):
            df_next_games_teams['score_1'].values[i] = 1.0 + np.log2(p)
            if df_next_games_teams['pred_home_result'].values[i] == 1:
                df_next_games_teams['score_2'].values[i] = 1.0 + np.log2(p)
                df_next_games_teams['score_3'].values[i] = 1.0 + np.log2(q)
            elif df_next_games_teams['pred_home_result'].values[i] == 0:
                df_next_games_teams['score_2'].values[i] = 1.0 + np.log2(1.0-p)
                df_next_games_teams['score_3'].values[i] = 1.0 + np.log2(1.0-q)
                
        elif df_next_games_teams['pred_home_result'].values[i] != df_next_games_teams['result'].values[i]:
            df_next_games_teams['score_1'].values[i] = 1.0 + np.log2(1.0 - p)

            if df_next_games_teams['pred_home_result'].values[i] == 1:
                df_next_games_teams['score_2'].values[i] = 1.0 + np.log2(1.0 - p)
                df_next_games_teams['score_3'].values[i] = 1.0 + np.log2(1.0 - q)
            elif df_next_games_teams['pred_home_result'].values[i] == 0:
                df_next_games_teams['score_2'].values[i] = 1.0 + np.log2(1.0-(1.0-p))
                df_next_games_teams['score_3'].values[i] = 1.0 + np.log2(1.0-(1.0-q))
            
            
    return accuracy, df_next_games_teams, features, afl_df