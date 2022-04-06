# Created by Maximilian Hofer in April 2022

# Import packages
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.io import export_png

# Set constants
K = 20

# Load data
df = pd.read_excel("../data/ipo_risk.xlsx", index_col=0)
print(df.head())
print(df.columns)

# Plot risk profile
firmName = 'Facebook'  # Set a firm name

riskTopics = df[df['Issuer'].str.contains(firmName)][['rf{}'.format(i) for i in range(K)]].values[0]
topicNames = [f'Risk Factor ({i})' for i in range(K)]
p = figure(x_range=topicNames,
           title='Risk profile for {}. Sum of risk topics is {}'.format(firmName, round(sum(riskTopics), 2)),
           y_range=(-1.2, 2.2),
           tools="",
           width=1200,
           height=600)
p.yaxis.axis_label = 'Risk topic loading (normalized)'
p.xaxis.major_label_orientation = 70
baseline = [0 for i in range(K + 1)]
p.line(range(K + 1), baseline, width=2, line_color='black', legend_label='Baseline', line_dash='dashed')
p.vbar(x=topicNames, top=riskTopics, width=0.5, legend_label='Risk topic loading', fill_color='black', line_color=None)
p.legend.location = "top_center"
export_png(p, filename=f"riskProfile_{firmName}.png")
