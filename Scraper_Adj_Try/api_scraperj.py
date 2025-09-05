import requests
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytz import timezone
import re


class MLB_Scrape:

    def __init__(self):
        # Initialize your class here if needed
        pass

    def get_sport_id(self):
        """
        Retrieves the list of sports from the MLB API and processes it into a Polars DataFrame.
        
        Returns:
        - df (pl.DataFrame): A DataFrame containing the sports information.
        """
        # Make API call to retrieve sports information
        response = requests.get(url='https://statsapi.mlb.com/api/v1/sports').json()
        
        # Convert the JSON response into a Polars DataFrame
        df = pl.DataFrame(response['sports'])
        
        return df

    def get_sport_id_check(self, sport_id: int = 1):
        """
        Checks if the provided sport ID exists in the list of sports retrieved from the MLB API.
        
        Parameters:
        - sport_id (int): The sport ID to check. Default is 1.
        
        Returns:
        - bool: True if the sport ID exists, False otherwise. If False, prints the available sport IDs.
        """
        # Retrieve the list of sports from the MLB API
        sport_id_df = self.get_sport_id()
        
        # Check if the provided sport ID exists in the DataFrame
        if sport_id not in sport_id_df['id']:
            print('Please Select a New Sport ID from the following')
            print(sport_id_df)
            return False
        
        return True


    def get_game_types(self):
        """
        Retrieves the different types of MLB games from the MLB API and processes them into a Polars DataFrame.
        
        Returns:
        - df (pl.DataFrame): A DataFrame containing the game types information.
        """
        # Make API call to retrieve game types information
        response = requests.get(url='https://statsapi.mlb.com/api/v1/gameTypes').json()
        
        # Convert the JSON response into a Polars DataFrame
        df = pl.DataFrame(response)
        
        return df

    def get_schedule(self,
                    year_input: list = [2024],
                    sport_id: list = [1],
                    game_type: list = ['R']):
        
        """
        Retrieves the schedule of baseball games based on the specified parameters.
        Parameters:
        - year_input (list): A list of years to filter the schedule. Default is [2024].
        - sport_id (list): A list of sport IDs to filter the schedule. Default is [1].
        - game_type (list): A list of game types to filter the schedule. Default is ['R'].
        Returns:
        - game_df (pandas.DataFrame): A DataFrame containing the game schedule information, including game ID, date, time, away team, home team, game state, venue ID, and venue name. If the schedule length is 0, it returns a message indicating that different parameters should be selected.
        """

        # Type checks
        if not isinstance(year_input, list) or not all(isinstance(year, int) for year in year_input):
            raise ValueError("year_input must be a list of integers.")
        if not isinstance(sport_id, list) or not all(isinstance(sid, int) for sid in sport_id):
            raise ValueError("sport_id must be a list of integers.")

        if not isinstance(game_type, list) or not all(isinstance(gt, str) for gt in game_type):
            raise ValueError("game_type must be a list of strings.")

        eastern = timezone('US/Eastern')

        # Convert input lists to comma-separated strings
        year_input_str = ','.join([str(x) for x in year_input])
        sport_id_str = ','.join([str(x) for x in sport_id])
        game_type_str = ','.join([str(x) for x in game_type])

        # Make API call to retrieve game schedule
        game_call = requests.get(url=f'https://statsapi.mlb.com/api/v1/schedule/?sportId={sport_id_str}&gameTypes={game_type_str}&season={year_input_str}&hydrate=lineup,players').json()
        try:
            # Extract relevant data from the API response
            game_list = [item for sublist in [[y['gamePk'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            time_list = [item for sublist in [[y['gameDate'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            date_list = [item for sublist in [[y['officialDate'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            away_team_list = [item for sublist in [[y['teams']['away']['team']['name'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            away_team_id_list = [item for sublist in [[y['teams']['away']['team']['id'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            home_team_list = [item for sublist in [[y['teams']['home']['team']['name'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            home_team_id_list = [item for sublist in [[y['teams']['home']['team']['id'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            state_list = [item for sublist in [[y['status']['codedGameState'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            venue_id = [item for sublist in [[y['venue']['id'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            venue_name = [item for sublist in [[y['venue']['name'] for y in x['games']] for x in game_call['dates']] for item in sublist]
            gameday_type = [item for sublist in [[y['gamedayType'] for y in x['games']] for x in game_call['dates']] for item in sublist]

            # Create a Polars DataFrame with the extracted data
            game_df = pl.DataFrame(data={'game_id': game_list,
                                        'time': time_list,
                                        'date': date_list,
                                        'away': away_team_list,
                                        'away_id': away_team_id_list,
                                        'home': home_team_list,
                                        'home_id': home_team_id_list,
                                        'state': state_list,
                                        'venue_id': venue_id,
                                        'venue_name': venue_name,
                                        'gameday_type': gameday_type})

            # Check if the DataFrame is empty
            if len(game_df) == 0:
                print('Schedule Length of 0, please select different parameters.')
                return None

            # Convert date and time columns to appropriate formats
            game_df = game_df.with_columns(
                game_df['date'].str.to_date(),
                game_df['time'].str.to_datetime().dt.convert_time_zone(eastern.zone).dt.strftime("%I:%M %p"))

            # Remove duplicate games and sort by date
            game_df = game_df.unique(subset='game_id').sort('date')

            # Check again if the DataFrame is empty after processing
            if len(game_df) == 0:
                print('Schedule Length of 0, please select different parameters.')
                return None
        except KeyError:
            print('No Data for Selected Parameters')
            return None
        
        return game_df

    def get_player_games_list(self, player_id: int, 
                              season: int, 
                              start_date: str = None, 
                              end_date: str = None, 
                              sport_id: int = 1, 
                              game_type: list = ['R'],
                              pitching: bool = True):
        """
        Retrieves a list of game IDs for a specific player in a given season.
        
        Parameters:
        - player_id (int): The ID of the player.
        - season (int): The season year for which to retrieve the game list.
        - start_date (str): The start date (YYYY-MM-DD) of the range (default is January 1st of the specified season).
        - end_date (str): The end date (YYYY-MM-DD) of the range (default is December 31st of the specified season).
        - sport_id (int): The ID of the sport for which to retrieve player data.
        - game_type (list): A list of game types to filter the schedule. Default is ['R'].
        - pitching (bool): Return pitching games.
        
        Returns:
        - player_game_list (list): A list of game IDs in which the player participated during the specified season.
        """
        # Set default start and end dates if not provided
        if not start_date:
            start_date = f'{season}-01-01'
        if not end_date:
            end_date = f'{season}-12-31'

        # Determine the group based on the pitching flag
        group = 'pitching' if pitching else 'hitting'

        # Validate date format
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if not date_pattern.match(start_date):
            raise ValueError(f"start_date {start_date} is not in YYYY-MM-DD format")
        if not date_pattern.match(end_date):
            raise ValueError(f"end_date {end_date} is not in YYYY-MM-DD format")

        # Convert game type list to a comma-separated string
        game_type_str = ','.join([str(x) for x in game_type])

        # Make API call to retrieve player game logs
        response = requests.get(url=f'http://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group={group},type=gameLog,season={season},startDate={start_date},endDate={end_date},sportId={sport_id},gameType=[{game_type_str}]),hydrations').json()
        
        # Check if stats are available in the response
        if 'stats' not in response['people'][0]:
            print(f'No {group} games found for player {player_id} in season {season}')
            return []

        # Extract game IDs from the API response
        player_game_list = [x['game']['gamePk'] for x in response['people'][0]['stats'][0]['splits']]
        
        return player_game_list
        
    def get_box_score(self, game_id):
        """
        Retrieves the box score data for a specific game ID and includes the game date.
        """
        # Fetch game data
        game_data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/boxscore").json()

        # Extract box score data and game date
        game_date = game_data['gameDate']  # Adjust based on the actual structure of the API response

        # Extract box score stats (example)
        box_score_data = {
            'game_id': game_id,
            'game_date': game_date,
            'player_id': [],
            'player_name': [],
            'runs': [],
            'hits': [],
            'at_bats': [],
            'strikeouts': [],
            'walks': [],
            'home_runs': [],
        }

        # Extract specific stats (adjust according to API response)
        for player in game_data['teams']['home']['players']:
            box_score_data['player_id'].append(player['id'])
            box_score_data['player_name'].append(player['fullName'])
            box_score_data['runs'].append(player['runs'])
            # Add other stats (hits, at_bats, etc.) similarly
        
        # Create a Polars DataFrame
        df = pl.DataFrame(box_score_data)
        return df
