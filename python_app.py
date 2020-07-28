import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_table
import dash_bootstrap_components as dbc
import base64

from dash.dependencies import Input, Output, State

# import uuid
# import os
# import flask

# external_stylesheets = [dbc.themes.LUX]
# app = dash.Dash(
#     __name__,
#     external_stylesheets=stylesheets
# )
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Data 
data = pd.read_csv('cselectric.csv')

# Month Vs Month
data['Month'] = [int(x.split('-')[1]) for x in list(data['Data'])]
data_april = data[data['Month'] == 4]
data_june = data[data['Month'] == 6]
data_may = data[data['Month'] == 5]

common_apr_jun = list(set(list(data_april['Query'])).intersection(set(list(data_june['Query']))))

common = {}
for x in common_apr_jun:
    common[x] = 'keep'

data_april['TO_keep'] = data_april['Query'].map(common)
data_june['TO_keep'] = data_june['Query'].map(common)

data_april = data_april[data_april['TO_keep'] == 'keep']
data_june = data_june[data_june['TO_keep'] == 'keep']

apr_pivot = pd.pivot_table(data_april, values = 'Impression', index=['Query'], columns = 'Data').reset_index()
jun_pivot = pd.pivot_table(data_june, values = 'Impression', index=['Query'], columns = 'Data').reset_index()

apr_pivot = apr_pivot.fillna(0)
jun_pivot = jun_pivot.fillna(0)

# Total = df['MyColumn'].sum()
apr_tot = []
jun_tot = []
for x in range(apr_pivot.shape[1]):
    if x == 0:
        pass
    else:
        col_a = apr_pivot.columns[x]
        col_j = jun_pivot.columns[x]
        apr_tot.append(apr_pivot[col_a].sum())
        jun_tot.append(jun_pivot[col_j].sum())
        

month = [x for x in range(1,31,1)]
month1 = apr_tot
month2 = jun_tot

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=month, y=month1, name='April-2020',
                         mode='lines+markers'))
fig.add_trace(go.Scatter(x=month, y=month2, name = 'June-2020'
                         ,mode='lines+markers'))


# Edit the layout
fig.update_layout(title='Month vs Month Comparision',
                   xaxis_title='Days',
                   yaxis_title='Impression',
                   plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text'])


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_subplot = make_subplots(rows=1, cols=2)

fig_subplot.add_trace(
    go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    row=1, col=1
)

fig_subplot.add_trace(
    go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    row=1, col=2
)

fig_subplot.update_layout(height=600, width=800, title_text="Side By Side Subplots")
# fig.show()


# app.layout = html.Div(children=[
#     html.H1(children='Comparision Tool'),

#     html.Div(children='''
#         Day-On-Day distribution analysis plot
#     '''),

#     dcc.Graph(
#         id='figure_top5',
#         figure=fig_pie5
#     )
# ])
apr_pivot['Row_sum'] = apr_pivot.sum(axis=1)
jun_pivot['Row_sum'] = jun_pivot.sum(axis=1)
apr_pie_5 = apr_pivot.sort_values(['Row_sum'], ascending=[False])
jun_pie_5 = jun_pivot.sort_values(['Row_sum'], ascending=[False])
apr_pie_5 = apr_pie_5[['Query','Row_sum']]
jun_pie_5 = jun_pie_5[['Query','Row_sum']]
apr_pie_5 = apr_pie_5[:5]
jun_pie_5 = jun_pie_5[:5]
apr_pie_5['Month'] = 'April-2020'
jun_pie_5['Month'] = 'June-2020'
pie_5 = pd.concat([apr_pie_5,jun_pie_5])



apr_pie_5['Month'] = 'April-2020'
jun_pie_5['Month'] = 'June-2020'
pie_5 = pd.concat([apr_pie_5,jun_pie_5])

fig_pie5 = px.sunburst(pie_5, path=['Month', 'Query'], values='Row_sum')
fig_pie5.update_layout(title='Top 5 Keywords Comparision',
                    font_color=colors['text'])

#################################################################################################################

apr_pie_10 = apr_pivot.sort_values(['Row_sum'], ascending=[False])
jun_pie_10 = jun_pivot.sort_values(['Row_sum'], ascending=[False])
apr_pie_10 = apr_pie_10[['Query','Row_sum']]
jun_pie_10 = jun_pie_10[['Query','Row_sum']]
apr_pie_10 = apr_pie_10[:10]
jun_pie_10 = jun_pie_10[:10]
apr_funnel = [apr_pivot['Row_sum'].sum(),apr_pie_10['Row_sum'].sum(), apr_pie_5['Row_sum'].sum()]
jun_funnel = [jun_pivot['Row_sum'].sum(),jun_pie_10['Row_sum'].sum(), jun_pie_5['Row_sum'].sum()]
grth_funnel  = []
for x in range(0,3):
    grth_funnel.append(jun_funnel[x] - apr_funnel[x])
# grth_funnel = [apr_pivot['Row_sum'].sum(),apr_pie_10['Row_sum'].sum(), apr_pie_5['Row_sum'].sum()]


from plotly import graph_objects as go

funnel_fig = go.Figure()

funnel_fig.add_trace(go.Funnel(
    name = 'April-2020',
    y = ["Total Impression", "Top 10 Imp", "Top 5 Imp"],
    x = apr_funnel,
    textinfo = "value+percent initial"))

funnel_fig.add_trace(go.Funnel(
    name = 'June-2020',
    orientation = "h",
    y = ["Total Impression", "Top 10 Imp", "Top 5 Imp"],
    x = jun_funnel,
    textposition = "inside",
    textinfo = "value+percent initial"))

funnel_fig.add_trace(go.Funnel(
    name = 'Growth',
    orientation = "h",
    y = ["Total Impression", "Top 10 Imp", "Top 5 Imp"],
    x = grth_funnel,
    textposition = "outside",
    textinfo = "value+percent initial"))

funnel_fig.update_layout(title='Growth Funnel',
                    font_color=colors['text'])

#########################################################################
data1 = apr_pivot[['Query','Row_sum']]
data2 = jun_pivot[['Query','Row_sum']]
data_table = pd.merge(data1, data2, on='Query')

data_table.columns = ['Query','Impression_April','Impression_June']

data_table['Growth'] = data_table['Impression_June'] - data_table['Impression_April']
data_table['Growth%'] = (data_table['Growth']/data_table['Impression_April'])*100
data_table['Growth%'] = data_table['Growth%'].round(2)

data_table_full = data_table.sort_values(['Growth'], ascending=[False])


data_table_gainer = data_table.sort_values(['Growth'], ascending=[False]).head(10)

# Biggest Loser 
data_table_loser = data_table.sort_values(['Growth'], ascending=[True]).head(10)


import plotly.graph_objects as go

fig_loser = go.Figure()


fig_loser.add_trace(
    go.Table(
        header=dict(
            values=['Query' ,'Impression_April' ,'Impression_June' ,'Growth' ,'Growth%'],
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[data_table_loser[k].tolist() for k in data_table_loser.columns],
            align = "left")
    )
)
fig_loser.update_layout(
    height=800,
    showlegend=False,
    title_text="Top 10 Losers",
)


fig_gainer = go.Figure()


fig_gainer.add_trace(
    go.Table(
        header=dict(
            values=['Query' ,'Impression_April' ,'Impression_June' ,'Growth' ,'Growth%'],
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[data_table_gainer[k].tolist() for k in data_table_gainer.columns],
            align = "left")
    )
)
fig_gainer.update_layout(
    height=800,
    showlegend=False,
    title_text="Top 10 Gainers",
)


fig_full_table = go.Figure()


fig_full_table.add_trace(
    go.Table(
        header=dict(
            values=['Query' ,'Impression_April' ,'Impression_June' ,'Growth' ,'Growth%'],
            # font=dict(size=10),
            align="left",
              line_color='darkslategray',
    fill_color='royalblue',
    # align=['left','center'],
    font=dict(color='white', size=12),
    height=40
        ),
        cells=dict(
            values=[data_table_full[k].tolist() for k in data_table_full.columns],
            # align = "left",
            line_color='darkslategray',
    fill=dict(color=['paleturquoise', 'white', 'white', 'white', 'white']),
    align=['left', 'center','center','center','center'],
    font_size=12)
    )
)


# cells=dict(
#     values=values,
#     line_color='darkslategray',
#     fill=dict(color=['paleturquoise', 'white']),
#     align=['left', 'center'],
#     font_size=12,
#     height=30)


fig_full_table.update_layout(
    height=1200,
    showlegend=False,
    title_text="Query Wise Comparison",
)





df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()



# dbc.Container([
#         dbc.Row([
#             dbc.Col(html.H1("COVID-19 in Singapore at a glance"), className="mb-2")
#         ]),
#         dbc.Row([
#             dbc.Col(html.H6(children='Visualising trends across the different stages of the COVID-19 outbreak in Singapore'), className="mb-4")
#         ]),

#         dbc.Row([
#             dbc.Col(dbc.Card(html.H3(children='Latest Update',
#                                      className="text-center text-light bg-dark"), body=True, color="dark")
#                     , className="mb-4")
#         ])



logo = go.Figure()

logo.layout.images = [dict(
        source="https://raw.githubusercontent.com/cldougl/plot_images/add_r_img/accuweather.jpeg",
        xref="paper", yref="paper",
        x=0.1, y=1.05,
        sizex=0.4, sizey=0.4,
        xanchor="center", yanchor="bottom"
      )]


image_filename = 'inde.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


app.layout = html.Div([ 


    html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    ]),


    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Growth Analyzer"), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Analyze business performance for different time frames'), className="mb-4")
        ])
        ]),




    html.Div([

    dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Day-on-Day Comparision',
                                     className="text-center text-light bg-dark"), body=True, color="dark")
                    , className="mb-4")
        ]),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
    ]),

    html.Div([

    html.Div([

        # dbc.Row([
        #     dbc.Col(dbc.Card(html.H3(children='Top 5 Keywords Comparision',
        #                              className="text-center text-light bg-dark"), body=True, color="dark")
        #             , className="mb-4")
        # ]),
           dcc.Graph(
        id='figure_top5',
        figure=fig_pie5
    )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    
    html.Div([

    
        # dbc.Row([
        #     dbc.Col(dbc.Card(html.H3(children='Growth Funnel',
        #                              className="text-center text-light bg-dark"), body=True, color="dark")
        #             , className="mb-4")
        # ]),
           dcc.Graph(
        id='figure_funnel',
        figure=funnel_fig
    )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}) ]),

    html.Div([

    html.Div([
           dcc.Graph(id='figure_gainer',
           figure=fig_gainer
    )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    
    html.Div([
           dcc.Graph(
        id='figure_loser',
        figure=fig_loser
    )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}) ]),

    html.Div([
          dcc.Graph(
        id='figure_subplot',
        figure=fig_subplot)
    ]),

    html.Div([
           dcc.Graph(
        id='figure_full_table',
        figure=fig_full_table
    )
    ], style={'display': 'inline-block', 'padding': '0 20'})


    # html.Div(dcc.Slider(
    #     id='crossfilter-year--slider',
    #     min=df['Year'].min(),
    #     max=df['Year'].max(),
    #     value=df['Year'].max(),
    #     marks={str(year): str(year) for year in df['Year'].unique()},
    #     step=None
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


# @app.callback(
#     dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
#     [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
#      dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
            )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='Year', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


# @app.callback(
#     dash.dependencies.Output('y-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#     dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#     dff = dff[dff['Indicator Name'] == yaxis_column_name]
#     return create_time_series(dff, axis_type, yaxis_column_name)


# fig.show()

# fig_gainer.write_html('first_gainer.html', auto_open=True)
# fig_loser.write_html('first_loser.html', auto_open=True)
# funnel_fig.write_html('first_funnel.html', auto_open=True)

if __name__ == '__main__':
    app.run_server(debug=True)