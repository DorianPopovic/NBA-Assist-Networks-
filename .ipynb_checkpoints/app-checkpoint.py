from dash import Dash, callback, html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import gunicorn                     #whilst your local machine's webserver doesn't need this, Heroku's linux webserver (i.e. dyno) does. I.e. This is your HTTP server
from whitenoise import WhiteNoise   #for serving static files on Heroku

import matplotlib.pyplot as plt
import networkx as nx
#import requests
#import json

import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'colab'
from plotly.offline import init_notebook_mode, iplot

# Dash app
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pickle


# Load saved data scraped from the web
link = 'https://raw.githubusercontent.com/DorianPopovic/NBA-Assist-Networks-/master/data.pkl'
teams_dict = pd.read_pickle(link)

# Initialize dash app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/nba-networks.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title = "NBA Assist Networks"


# Function that outputs the interactive plot
def plot_NBA_assist_network(team, layout):
    
        data = teams_dict[team][0]
        labels = list(data.columns)
        node_size=[300*i/sum(teams_dict[team][1]) for i in teams_dict[team][1]]

        G = nx.from_numpy_matrix(np.array(data.values, dtype=int), parallel_edges=True)
        
        if layout=="circular":
            pos = nx.circular_layout(G)
        elif layout=="random":
            pos = nx.random_layout(G)
        elif layout=="kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        
            
        nx.set_node_attributes(G, pos, 'pos')

        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())

        weights=[0.09*i for i in list(weights)]

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        ed_x = []
        ed_y = []
        
        for i in range(0, len(edge_x), 3):
          # Get list of 3 consecutive elements 
          ed_x.append([edge_x[i], edge_x[i+1], edge_x[i+2]])
          ed_y.append([edge_y[i], edge_y[i+1], edge_y[i+2]])

        edge_traces={}
        for i in range(len(weights)):
          edge_traces['trace_' + str(i)] = go.Scatter(
              x=ed_x[i], y=ed_y[i],
              line=dict(width=weights[i], color='#888'),
              hoverinfo='text',
              mode='lines')

        node_x = []
        node_y = []

        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=labels,
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Blackbody',
                reversescale=True,
                color=[],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Player connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []

        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('# of connections: '+str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = [', '.join(a) for a in zip(labels, node_text)]

        plot_data=list(edge_traces.values())
        plot_data.append(node_trace)

        fig = go.Figure(data=plot_data,
                    layout=go.Layout(
                        template = 'plotly_white',
                        title="Assists Network for: " + team,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text = "NBA Assist Networks, Dorian Popovic",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.update_layout(
            autosize=False,
            width=700,
            height=600)
        
        fig.show()
        return fig
    
    
# Helper function for plotting the degree distribution of a Graph
def plot_degree_distribution(team):
    
    data = teams_dict[team][0]
    G = nx.from_numpy_matrix(np.array(data.values, dtype=int), parallel_edges=True)
    degrees = {}
    for node in G.nodes():
        degree = G.degree(node)
        if degree not in degrees:
            degrees[degree] = 0
        degrees[degree] += 1
    sorted_degree = sorted(degrees.items())
    deg = [k for (k,v) in sorted_degree]
    cnt = [v for (k,v) in sorted_degree]
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, color='b', alpha=0.75)
    plt.title("Assist Network Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)
    
    return ax
    
# Render matplotlib barplot to plotly one
def bar_to_plotly(ax: mpl.axes.Axes) -> go.Figure:
    x, w, h, c = [], [], [], []
    rects = [rect for rect in ax.get_children()
             if isinstance(rect, mpl.patches.Rectangle)][:-1]
    for rect in rects:
        x.append(rect.get_x())
        w.append(rect.get_width())
        h.append(rect.get_height())
        c.append(f"rgb({','.join(map(str, np.array(rect.get_facecolor()) * 255))})")
    offset = w[0] / 2.
    ticks = {"vals": [], "text": []}
    for rect in ax.get_xticklabels():
        ticks["vals"].append(rect.get_position()[0] - offset)
        ticks["text"].append(rect.get_text())
    f = go.Figure([go.Bar(x=x, y=h, width=w, marker_color=c, opacity=0.5)],
                  layout=go.Layout(barmode="group"))
    f.update_layout(
        height=333,
        template='plotly_white',
        title=ax.get_title(),
        xaxis={
            "title_text": ax.get_xlabel(),
            "tickmode": "array",
            "range": [ax.get_xlim()[0], ax.get_xlim()[1]-w[0]],
            "tickvals": ticks["vals"],
            "ticktext": ticks["text"]
        },
        yaxis={
            "title_text": ax.get_ylabel(),
            "range": ax.get_ylim()
        }
    )
    
    return f


# Modal
# Modal
with open("explanations.md", "r") as f:
    howto_md = f.read()

"""""
modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)
"""""

button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/DorianPopovic",
    id="gh-link",
    style={"text-transform": "none"},
)

# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src=app.get_asset_url("basketball.png"),
                            height="60px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("NBA Assist Networks"),
                                    html.P("Leveraging graph theory to connect NBA players"),
                                ],
                                id="app-title",
                            )
                        ],
                        md=True,
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(button_github),
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                            #modal_overlay,
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    color="dark",
    sticky="top",
)
    
    
# Description
# Description
description = dbc.Col(
    [
        dbc.Card(
            id="description-card",
            children=[
                dbc.CardHeader("Explanation"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Img(
                                            src="assets/logo.png",
                                            width="200px",
                                        )
                                    ],
                                    md="auto",
                                ),
                                dbc.Col(
                                    html.P(
                                        "This is the webpage where the NBA Assist Network application is deployed. "
                                        "This application enables the visualization of 2021-22 regular season NBA teams through their assist network graphs! "
                                        "But what is an assist network graph? "
                                        "For each team, players are connected through the number of assists that happened between them (undirected graph). "
                                        "The size of a player's node is directly proportional to the percentage of the team's total assists that he contributed to. "
                                        "The width/strength of the edges between the players is proportionnal to how many times during the seasons an assist occured between them (undirected, an assist from player A to player B is counted as the same as an assist from player B to player A). "
                                        "The number of connection tells us to how many players a given player is connected (either made an assist to them or received an assist from them during the season). "
                                        "Finally the color of a player's node also reflects his number of connections. "
                                        "Below you can select which team's assist network you want to visualize and hover your cursor over a player's node to get more information. "
                                    ),
                                    md=True,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
        )
    ],
    md=12,
)

# Network Visualization
network = [
    dbc.Card(
        id="network-card",
        children=[
            dbc.CardHeader("Viewer"),
            dbc.CardBody(
                [
                    dcc.Graph(id = 'NBA_plot')
                ]
            )
        ]
    )
]

# Degree distribution plot
degree = [
    dbc.Card(
        id="degree-card",
        children=[
            dbc.CardHeader("Graph Degree Distribution"),
            dbc.CardBody(
                [
                    dcc.Graph(id = 'Degree_plot')
                ],
                style={"width": "100%", "height": "100%"}
            )
        ]
    )
]

# sidebar
sidebar = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Tools"),
            dbc.CardBody(
                [
                     html.H6("Choose your NBA team:", className="card-title"),
                     dcc.Dropdown( id = 'teams_dropdown',
                     options = [{'label':'Atlanta Hawks', 'value':'ATL' },
                                {'label': 'Brooklyn Nets', 'value':'BKN'},
                                {'label': 'Boston Celtics', 'value':'BOS'},
                                {'label': 'Charlotte Hornets', 'value':'CHA'},
                                {'label': 'Chicago Bulls', 'value':'CHI'},
                                {'label': 'Cleveland Cavaliers', 'value':'CLE'},
                                {'label': 'Dallas Mavericks', 'value':'DAL'},
                                {'label': 'Denver Nuggets', 'value':'DEN'},
                                {'label': 'Detroit Pistons', 'value':'DET'},
                                {'label': 'Golden State Warriors', 'value':'GSW'},
                                {'label': 'Houston Rockets', 'value':'HOU'},
                                {'label': 'Indiana Pacers', 'value':'IND'},
                                {'label': 'Los Angeles Clippers', 'value':'LAC'},
                                {'label': 'Los Angeles Lakers', 'value':'LAL'},
                                {'label': 'Memphis Grizzlies', 'value':'MEM'},
                                {'label': 'Miami Heat', 'value':'MIA'},
                                {'label': 'Milwaukee Bucks', 'value':'MIL'},
                                {'label': 'Minnesota Timberwolves', 'value':'MIN'},
                                {'label': 'New Orleans Pelicans', 'value':'NOP'},
                                {'label': 'New York Knicks', 'value':'NYK'},
                                {'label': 'Oklahoma City Thunder', 'value':'OKC'},
                                {'label': 'Orlando Magic', 'value':'ORL'},
                                {'label': 'Philadelphia 76ers', 'value':'PHI'},
                                {'label': 'Phoenix Suns', 'value':'PHX'},
                                {'label': 'Portland Trail Blazers', 'value':'POR'},
                                {'label': 'Sacramento Kings', 'value':'SAC'},
                                {'label': 'San Antonio Spurs', 'value':'SAS'},
                                {'label': 'Toronto Raptors', 'value':'TOR'},
                                {'label': 'Utah Jazz', 'value':'UTA'},
                                {'label': 'Washington Wizards', 'value':'WAS'},],
                     value = 'ATL',
                     style={"width": "80%"}
                    ),
                    html.Hr(),
                    html.H6("Select the network layout", className="card-title"),
                    dcc.Dropdown( id = 'layout_dropdown',
                     options = [{'label':'Circular', 'value':'circular' },
                                {'label': 'Random', 'value':'random'},
                                {'label': 'Kamada Kawai', 'value':'kamada_kawai'},],
                     value = 'circular',
                     style={"width": "80%"}
                    ),
                ]
            ),
            
        ],
    ),
]
    
app.layout =html.Div([

        header,
        dbc.Container(
            [
                dbc.Row(description),
                dbc.Row(
                    id="app-content",
                    children=[dbc.Col(network, md=7),
                              dbc.Col([dbc.Row(sidebar), dbc.Row(degree)], md=5)],
                )
            ],
            fluid=True,
        ),
    ]
                     )


@app.callback(Output(component_id='NBA_plot', component_property= 'figure'),
              [Input(component_id='teams_dropdown', component_property= 'value'),
               Input(component_id='layout_dropdown', component_property= 'value')])
def graph_update(teams_dropdown_value, layouts_dropdown_value):
    return plot_NBA_assist_network(teams_dropdown_value, layouts_dropdown_value), bar_to_plotly(plot_degree_distribution(teams_dropdown_value))



# Run flask app
if __name__ == "__main__": 
    app.run_server(debug=False, host='0.0.0.0', port=8050)
