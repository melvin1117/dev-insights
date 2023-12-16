from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from assets.constants import PROFICIENCY_COLORS, DASHBOARD_CHART_THEME, PROFICIENCY, GITHUB
from database.session import Session
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

class DeveloperDashboard:
    def __init__(self):
        self.proficiency_levels = PROFICIENCY
        # Initialize Dash app
        self.app = Dash(__name__, external_stylesheets=['./assets/style.css'])

        # Fetch and preprocess data
        self.user_df = self.fetch_users()
        self.repo_df = self.fetch_repo()
        
        if not ((self.user_df is not None and not self.user_df.empty) or (self.repo_df is not None and not self.repo_df.empty)):
            self.app.layout = self.create_error_layout()
        else:
            self.preprocess_data()

            # Set up initial visualizations
            self.fig_user_count = self.create_user_count_chart()
            self.fig_proficiency = self.create_proficiency_chart()
            self.fig_mean_yoe = self.create_mean_yoe_chart()
            self.fig_world_map = self.create_world_map_chart()

            # Set up the layout
            self.app.layout = self.create_layout()

    def fetch_users(self) -> pd.DataFrame:
        """Fetch user data from local MongoDB."""
        try:
            with Session() as session:
                try:
                    cursor = session[GITHUB['user']].find({})
                    result_df = pd.DataFrame(list(cursor))
                    
                    # Check if the result is empty
                    if result_df.empty:
                        # Call another function in case of an empty result
                        return self.fetch_users_fallback()
                    
                    return result_df
                except Exception as e:
                    # Log the error and call the fallback function
                    logger.error(f"Error fetching users: {e}")
                    return self.fetch_users_fallback()
        except Exception as e:
            # Log the error and call the fallback function
            logger.error(f"Error fetching users: {e}")
            return self.fetch_users_fallback()

    def fetch_users_fallback(self) -> pd.DataFrame:
        """Fetch user data from MongoDB from the server."""
        try:
            with Session(server_host=True) as session:
                try:
                    logger.info("Attempting to fallback. Fetching users from server")
                    cursor = session[GITHUB['user']].find({})
                    return pd.DataFrame(list(cursor))
                except Exception as e:
                    logger.error(f"Error fetching users from server: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching users from server: {e}")
            return None

    def fetch_repo(self) -> pd.DataFrame:
        """Fetch repo data from local MongoDB."""
        try:
            with Session() as session:
                try:
                    cursor = session[GITHUB['repo']].find({})
                    result_df = pd.DataFrame(list(cursor))
                    
                    # Check if the result is empty
                    if result_df.empty:
                        # Call another function in case of an empty result
                        return self.fetch_repo_fallback()
                    
                    return result_df
                except Exception as e:
                    # Log the error and call the fallback function
                    logger.error(f"Error fetching repo: {e}")
                    return self.fetch_repo_fallback()
        except Exception as e:
            # Log the error and call the fallback function
            logger.error(f"Error fetching repo: {e}")
            return self.fetch_repo_fallback()

    def fetch_repo_fallback(self) -> pd.DataFrame:
        """Fetch repo data from MongoDB the server."""
        try:
            with Session(server_host=True) as session:
                try:
                    logger.info("Attempting to fallback. Fetching repo from server")
                    cursor = session[GITHUB['repo']].find({})
                    return pd.DataFrame(list(cursor))
                except Exception as e:
                    logger.error(f"Error fetching repos from server: {e}")
                    return None
        except Exception as e:
                logger.error(f"Error fetching repos from server: {e}")
                return None

    def preprocess_repo_data(self):
        """Preprocess data for repository analysis."""
        self.repo_df['created_at'] = pd.to_datetime(self.repo_df['created_at']).dt.tz_localize(None)
        self.repo_df['year'] = self.repo_df['created_at'].dt.year
        self.calculate_mean_std_n_rating_by_language()

    def preprocess_data(self):
        """Preprocess data for analysis."""
        self.user_df['created_at'] = pd.to_datetime(self.user_df['created_at']).dt.tz_localize(None)
        self.user_df['year'] = self.user_df['created_at'].dt.year
        self.user_df['yoe'] = (pd.to_datetime('now') - self.user_df['created_at']).dt.days / 365.25
        self.calculate_proficiency_percentages()
        self.calculate_mean_yoe_by_language()
        self.heatmap_data = self.format_data_for_heatmap()
        self.figures_by_language = {}
        for language in self.languages:
            self.figures_by_language[language] = self.create_language_chart(language)
        self.preprocess_repo_data()

    def calculate_proficiency_percentages(self):
        """Calculate proficiency percentages for each language."""
        self.proficiency_percentages = {}
        self.total_users_by_lang = {}
        self.languages = self.user_df['n_rating'].apply(lambda x: list(x.keys())).explode().unique()
        for language in self.languages:
            proficiency_counts = self.user_df['proficiency'].apply(lambda x: x.get(language, np.nan)).value_counts()
            total_users = proficiency_counts.sum()
            percentages = {}
            self.total_users_by_lang[language] = total_users
            for proficiency_level in self.proficiency_levels:
                count = proficiency_counts.get(proficiency_level, 0)
                percentage = (count / total_users) * 100 if total_users > 0 else 0
                percentages[proficiency_level] = round(percentage, 2)

            self.proficiency_percentages[language] = percentages
        self.sorted_total_users_by_lang = dict(
            sorted(self.total_users_by_lang.items(), key=lambda item: item[1], reverse=True)
        )
        # make all same order
        sorted_keys = sorted(self.proficiency_percentages, key=lambda x: list(self.sorted_total_users_by_lang.keys()).index(x))
        self.proficiency_percentages = {k: self.proficiency_percentages[k] for k in sorted_keys}

    def calculate_mean_yoe_by_language(self):
        """Calculate mean years of experience for each language."""
        self.mean_yoe_by_lang = {}
        for language in self.languages:
            language_rows = self.user_df[self.user_df['proficiency'].apply(lambda x: language in x.keys())]
            mean_yoe = language_rows['yoe'].mean()
            self.mean_yoe_by_lang[language] = round(mean_yoe / 3, 2)

        # Sort mean YOE by language in descending order
        self.sorted_mean_yoe_by_lang = dict(
            sorted(self.mean_yoe_by_lang.items(), key=lambda item: item[1], reverse=True)
        )

    def create_repo_count_chart(self):
        """Create a horizontal bar chart for the number of repositories by programming language."""
        sorted_repo_counts = self.repo_df['language'].value_counts().sort_values(ascending=True)

        fig_repo_count = px.bar(
            x=sorted_repo_counts.values,
            y=sorted_repo_counts.index,
            orientation='h',
            labels={'x': 'Repositories', 'y': 'Programming Language'},
            title=f'Repo Pulse: Counting Code Containers by Language',
            color_discrete_sequence=DASHBOARD_CHART_THEME
        )
        return fig_repo_count

    def calculate_mean_std_n_rating_by_language(self):
        """Calculate mean and standard deviation of n_rating for repositories grouped by language."""
        self.mean_std_n_rating_by_lang = self.repo_df.groupby('language')['n_rating'].agg(['mean', 'std']).reset_index()

    def create_n_rating_chart(self):
        """Create a bar chart for the mean and standard deviation of n_rating for repositories grouped by language."""
        return go.Figure(data=[
            go.Bar(
                x=self.mean_std_n_rating_by_lang['language'],
                y=self.mean_std_n_rating_by_lang['mean'],
                name='Mean rating',
                marker_color=DASHBOARD_CHART_THEME[0]
            ),
            go.Bar(
                x=self.mean_std_n_rating_by_lang['language'],
                y=self.mean_std_n_rating_by_lang['std'],
                name='Std dvd rating',
                marker_color=DASHBOARD_CHART_THEME[1]
            )
        ]).update_layout(
            xaxis=dict(title='Programming Language'),
            yaxis=dict(title='Value'),
            title='Rating Realms: Mean & Deviation Across Languages',
            barmode='group',
            colorway=DASHBOARD_CHART_THEME
        )

    def create_user_count_chart(self):
        """Create a bar chart for the number of users by programming language."""
        return px.bar(
            self.user_df,
            x=list(self.sorted_total_users_by_lang.keys()),
            y=list(self.sorted_total_users_by_lang.values()),
            labels={'y': 'Users', 'x': 'Programming Language'},
            title='Programming Language Popularity: User Count Breakdown',
            color_discrete_sequence=DASHBOARD_CHART_THEME
        )

    def create_proficiency_chart(self):
        """Create a stacked bar chart for proficiency levels by language."""
        fig_proficiency = go.Figure()
        for proficiency_level in self.proficiency_levels:
            fig_proficiency.add_trace(go.Bar(
                x=list(self.proficiency_percentages.keys()),
                y=[values.get(proficiency_level, 0) for values in self.proficiency_percentages.values()],
                name=proficiency_level
            ))

        fig_proficiency.update_layout(
            xaxis=dict(title='Programming Language'),
            yaxis=dict(title='Percentage'),
            title='Skill Spectrum: Proficiency Levels Across Languages',
            barmode='stack',
            colorway=DASHBOARD_CHART_THEME
        )
        return fig_proficiency

    def calculate_proficiency_counts_by_language(self):
        """Calculate proficiency level counts for each language."""
        proficiency_counts_by_lang = {}
        for language in self.languages:
            proficiency_counts = self.user_df['proficiency'].apply(lambda x: x.get(language, np.nan)).value_counts()
            proficiency_counts_by_lang[language] = proficiency_counts

        return proficiency_counts_by_lang

    def create_pareto_chart(self, language):
        """Create a Pareto chart for proficiency levels in a specific language."""
        proficiency_percentages = self.proficiency_percentages[language]
        
        # Extract keys and values from the proficiency dictionary
        proficiency_levels = list(proficiency_percentages.keys())
        percentages = list(proficiency_percentages.values())

        # Calculate cumulative percentage
        cumulative_percentage = [sum(percentages[:i + 1]) for i in range(len(percentages))]

        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for proficiency levels
        fig_pareto.add_trace(go.Bar(
            x=proficiency_levels,
            y=percentages,
            name='Proficiency Levels',
            marker_color=DASHBOARD_CHART_THEME[0]
        ))

        # Line chart for cumulative percentage
        fig_pareto.add_trace(go.Scatter(
            x=proficiency_levels,
            y=cumulative_percentage,
            name='Cumulative Percentage',
            mode='lines+markers',
            line=dict(color='#FFB703'),
            marker=dict(color='#FFB703', size=8),
            yaxis='y2'
        ))

        fig_pareto.update_layout(
            xaxis=dict(title='Proficiency Level'),
            yaxis=dict(title='Percentage'),
            yaxis2=dict(
                title='Cumulative Percentage',
                overlaying='y',
                side='right',
                showgrid=False,
                showline=False,
                showticklabels=True,
            ),
            title=f'Pareto Proficiency: Dominance in  {language}',
            barmode='group',
            colorway=[DASHBOARD_CHART_THEME[0]],
        )

        return fig_pareto

    def format_data_for_heatmap(self):
        """Format data for heatmap."""
        heatmap_data = {}
        for language in self.languages:
            language_data = {'Beginner': [], 'Intermediate': [], 'Expert': []}
            for index, row in self.user_df.iterrows():
                proficiency = row['proficiency'].get(language, None)
                if proficiency:
                    location = row['loc']
                    language_data[proficiency].append({
                        'lat': location.get('lat'),
                        'lng': location.get('lng'),
                        'formatted_address': location.get('formatted_address')
                    })
            heatmap_data[language] = language_data
        return heatmap_data

    def create_mean_yoe_chart(self):
        """Create a bar chart for mean years of experience by programming language."""
        return px.bar(
            x=list(self.sorted_mean_yoe_by_lang.keys()),
            y=list(self.sorted_mean_yoe_by_lang.values()),
            labels={'y': 'Mean Years of Experience', 'x': 'Programming Language'},
            title='Years in the Code: Mean Experience by Language',
            color_discrete_sequence=DASHBOARD_CHART_THEME
        )

    def create_world_map_chart(self):
        """Create a world map chart for developer locations."""
        # Create new columns for 'lat', 'lng', and 'formatted_address'
        self.user_df['lat'] = self.user_df['loc'].apply(lambda x: x.get('lat') if x else None)
        self.user_df['lng'] = self.user_df['loc'].apply(lambda x: x.get('lng') if x else None)
        self.user_df['formatted_address'] = self.user_df['loc'].apply(lambda x: x.get('formatted_address') if x else None)

        # Drop rows with missing lat, lng, or formatted_address
        self.user_df = self.user_df.dropna(subset=['lat', 'lng', 'formatted_address'])

        location_counts = self.user_df.groupby(['lat', 'lng', 'formatted_address']).size().reset_index(name='user_count')

        # Create a bubble map
        fig_world_map = px.scatter_geo(
            location_counts,
            lat='lat',
            lon='lng',
            size='user_count',  # Use 'user_count' as the bubble size
            size_max=30,  # Set the maximum bubble size
            title='Developer Atlas: Unraveling Worldwide Talent',
            template='plotly',
            hover_name='formatted_address',  # Add 'formatted_address' to the tooltip
            hover_data={'formatted_address': True, 'user_count': True},  # Include 'formatted_address' and 'user_count' in the tooltip
            color_discrete_sequence=DASHBOARD_CHART_THEME
        )

        # Adjust map layout for better visibility of bubbles
        fig_world_map.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
        )
        return fig_world_map

    def update_world_map(self, selected_year):
        """Update world map based on selected year."""
        # Filter data based on selected year
        filtered_df = self.user_df[self.user_df['year'] <= selected_year]

        # Update bubble map data
        location_counts = filtered_df.groupby(['lat', 'lng', 'formatted_address']).size().reset_index(name='user_count')

        # Check if 'year' column exists in location_counts
        if 'year' not in location_counts.columns:
            location_counts['year'] = selected_year

        # Update figure data
        self.fig_world_map.update_traces(
            lat=location_counts['lat'],
            lon=location_counts['lng'],
            customdata=location_counts[['formatted_address', 'user_count', 'year']],
            hovertemplate='Address: %{customdata[0]}<br>Users Count: %{customdata[1]}<br>Year: %{customdata[2]}'
        )

    def create_language_chart(self, language):
        """Create a map chart for proficiency levels by language."""
        language_data = self.heatmap_data[language]

        data = []
        all_sizes = []

        # Iterate through proficiency levels to collect all sizes
        for proficiency_level, locations in language_data.items():
            if locations:
                # Dictionary to store counts and formatted_address for each (lat, lng) pair
                lat_lng_info = {}
                for location in locations:
                    lat_lng = (location['lat'], location['lng'])
                    formatted_address = location.get('formatted_address', '')
                    if lat_lng not in lat_lng_info:
                        lat_lng_info[lat_lng] = {'count': 0, 'formatted_address': formatted_address}
                    lat_lng_info[lat_lng]['count'] += 1

                # Separate latitudes, longitudes, sizes, and formatted_addresses
                latitudes, longitudes, sizes, formatted_addresses = zip(
                    *[(lat, lng, info['count'], info['formatted_address']) for (lat, lng), info in lat_lng_info.items()])

                all_sizes.extend(sizes)

                # Create a single trace for the proficiency level
                data.append(go.Scattergeo(
                    lat=latitudes,
                    lon=longitudes,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        sizemode='diameter',
                        color=PROFICIENCY_COLORS[proficiency_level],
                        opacity=0.7,
                    ),
                    name=f"{proficiency_level} ({len(locations)})",
                    text=[f"Size: {size}<br>Address: {formatted_address}" for size, formatted_address in zip(sizes, formatted_addresses)],
                    hoverinfo='text',  # Show only the text in the tooltip
                ))

        # Calculate the scaling factor based on the maximum size across all proficiency levels
        max_size = 30  # You can adjust this value based on your preference
        scaling_factor = max_size / max(all_sizes)

        # Apply the scaling factor to all sizes
        for trace in data:
            trace.marker.size = [size * scaling_factor for size in trace.marker.size]

        fig_language_map = go.Figure(data=data)

        fig_language_map.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
        )

            # Update layout to increase map size
        fig_language_map.update_layout(
            geo=dict(
                domain=dict(x=[0, 1], y=[0, 1]),  # Adjust the y-values to control the vertical size
                resolution=50,
                showcoastlines=True,
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            ),
            height=650,
            geo2=dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                resolution=50,
                showcoastlines=True,
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            ),
            legend=dict(
                font=dict(
                    size=16  # Adjust the size as needed
                ),
                itemsizing= 'constant'
            )
        )

        return fig_language_map

    def create_layout(self):
        """Create the layout for the Dash app."""
        min_year = self.user_df['year'].min()
        max_year = self.user_df['year'].max()

        # Add Tabs component to the layout
        tabs = dcc.Tabs(id='language-tabs', value=self.languages[0], children=[
            dcc.Tab(label=language, value=language) for language in self.languages
        ])

        # Add a placeholder for the selected tab content
        tab_content = html.Div(id='language-content')

        # Add a styled divider and text before the last two graphs
        divider_text = html.Div([
            html.H2("Additional Repositories Information", style={'text-align': 'center', 'margin': '100px 0 10px 0', 'color': '#555'}),
            html.P("Here you can provide additional information or context about the github repository data.", style={'text-align': 'center','margin-bottom': '100px', 'color': '#777'}),
        ])

        return html.Div(children=[
            html.Nav(children=[
                html.H1(children='Dev Insights Dashboard', style={'flex': '1', 'textAlign': 'center', 'color': 'white'}),
            ], style={'background-color': '#333', 'padding': '10px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

            # Horizontal bar chart for the number of users by programming language
            dcc.Graph(id='user-count-chart', figure=self.fig_user_count, style={'margin': '20px'}),

            # Stacked bar chart for proficiency levels by language
            dcc.Graph(id='proficiency-chart', figure=self.fig_proficiency, style={'margin': '20px'}),

            # Bar chart for mean years of experience
            dcc.Graph(id='mean-yoe-chart', figure=self.fig_mean_yoe, style={'margin': '20px'}),

            # Div to group the world map chart and the year slider
            html.Div(children=[
                # World map chart for developer locations
                dcc.Graph(id='world-map-chart', figure=self.fig_world_map, style={'width': '100%', 'height': '700px'}),
                # Year slider
                html.Div(children=[
                    dcc.Slider(
                        id='year-slider',
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=max_year,
                        marks={str(year): str(year) for year in range(min_year, max_year + 1)},
                    ),
                ], style={'width': '50%', 'margin-bottom': '10px', 'padding-bottom': '15px', 'margin-top': '-30px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'},),
            ], style={'background': '#FFF', 'margin': '20px'}),
            # Tabs for language-specific charts
            html.Div(children=[
                tabs,
                tab_content,
            ], style={'background': '#FFF', 'margin': '20px'}),

            # Pareto chart for proficiency levels
            dcc.Graph(id='pareto-chart', figure=self.create_pareto_chart(self.languages[0]), style={'margin': '20px'}),

            # Additional Information Divider and Text
            divider_text,
            # Horizontal bar chart for the number of repositories by programming language
            dcc.Graph(id='repo-count-chart', figure=self.create_repo_count_chart(), style={'margin': '20px'}),

            # Bar chart for the mean and standard deviation of n_rating for repositories
            dcc.Graph(id='n-rating-chart', figure=self.create_n_rating_chart(), style={'margin': '20px'}),

        ], style={'font-family': 'Arial', 'width': '100%', 'background-color': '#f4f4f4', 'margin': '0', 'padding': '0'})

    def create_error_layout(self):
        """Displays error layout
        """
        return html.Div(
            [
                html.H1("No Data Found", style={"textAlign": "center", "margin-bottom": "10px"}),
                html.P("Unable to fetch data from the database and from the server. Try running the steps mentioned in the readme and try again.", style={"textAlign": "center"}),
            ],
            style={"font-family": "Arial", "height": "100vh", "display": "flex", "flexDirection": "column", "justifyContent": "center"},
        )
    
    def run_app(self):
        """Run the Dash app."""
        try:
            self.app.run_server(debug=False, host='0.0.0.0')
        except Exception as e:
            logger.error(f"Error running the dashboard app: {e}")
            raise

# Create an instance of DeveloperDashboard
dashboard = DeveloperDashboard()

@dashboard.app.callback(
    [Output('language-content', 'children'),
     Output('world-map-chart', 'figure'),
     Output('pareto-chart', 'figure')],
    [Input('language-tabs', 'value'),
     Input('year-slider', 'value')]
)
def update_tab_and_world_map(selected_language, selected_year):
    """Callback to update tab content, world map, and Pareto chart."""
    selected_figure = dashboard.figures_by_language.get(selected_language, None)
    tab_content = dcc.Graph(figure=selected_figure) if selected_figure else None

    # Update the world map based on the selected year
    dashboard.update_world_map(selected_year)

    # Update Pareto chart for proficiency levels
    pareto_chart = dashboard.create_pareto_chart(selected_language)

    return tab_content, dashboard.fig_world_map, pareto_chart


# Run the Dash app
if __name__ == '__main__':
    dashboard.run_app()
