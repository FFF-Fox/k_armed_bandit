from bokeh.plotting import figure, show
from bokeh.palettes import Spectral4

def plot_lines(lines, labels, title, x_label, y_label):
    """ Plots multiple lines on the same bokeh plot.
        Attributes:
        lines    :  List of equal length arrays to be plotted.
        labels   :  The labels for each individual line.
        title    :  The title of the plot.
        x_label  :  The label of x axis.
        y_label  : The label of y axis.
    """
    p = figure(plot_height=400, plot_width=650)

    x = range(len(lines[0]))

    for i in range(len(lines)):
        p.line(x,lines[i], legend=labels[i], color=Spectral4[i])


    # Legend
    p.legend.location = "bottom_right"
    p.legend.click_policy="hide"

    # Background color
    p.background_fill_color = "beige"
    p.background_fill_alpha = 0.5

    # Titles and labels
    p.title.text = title
    p.title.text_font_size = "1em"

    p.xaxis.axis_label = x_label
    p.xaxis.axis_label_text_color = "#aa6666"
    p.xaxis.axis_label_text_font_size = "0.9em"

    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_style = "italic"
    p.yaxis.axis_label_text_font_size = "0.9em"

    # Grid lines
    p.xgrid.grid_line_color = None

    p.ygrid.grid_line_alpha = 0.99
    p.ygrid.grid_line_dash = [6,3]

    show(p)
